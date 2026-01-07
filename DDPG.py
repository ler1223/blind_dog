import torch
import torch.nn as nn
import Evolution
import Visualizer
from collections import deque, namedtuple
from copy import deepcopy
import random

Experience = namedtuple('Experience',
                       ('state', 'action', 'reward', 'next_state'))


class GaussianNoise:
    def __init__(self, action_dim, sigma=0.2, sigma_min=0.01, sigma_decay=0.999):
        self.action_dim = action_dim
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay

    def noise(self):
        noise = torch.randn(self.action_dim) * self.sigma
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)
        return noise


class ReplayBuffer:
    def __init__(self, capacity, lstm=False):
        self.buffer = deque(maxlen=capacity)
        self.lstm = lstm

    def push(self, state, action, reward, next_state):
        self.buffer.append(Experience(state, action, reward, next_state))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        if self.lstm:
            padded_states = torch.nn.utils.rnn.pad_sequence(
                [e.state for e in experiences],
                batch_first=True,
                padding_value=0.0
            )
            states = padded_states
        else:
            states = torch.cat([e.state for e in experiences])
        actions = torch.cat([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.cat([e.next_state for e in experiences])

        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class LSTMEncoder(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_size, batch_first=True, num_layers=1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        x = lstm_out[:, -1, :]
        return x


class ActorNetwork(Evolution.EvolutionaryNetwork):
    def __init__(self, encoder, input_size=8, hidden_size=64, output_size=2, lstm=False):
        super().__init__()
        self.lstm = lstm
        self.output_size = output_size
        if lstm:
            self.encoder = encoder

            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size),
                nn.Tanh()
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size),
                nn.Tanh()
            )

    def forward(self, x):
        if self.lstm:
            x = self.encoder(x)
            return self.net(x)
        else:
            return self.net(x).view(-1, self.output_size)


class CriticNetwork(nn.Module):
    def __init__(self, encoder, state_size=42, action_size=2, hidden_size=64, output_size=1, lstm=False):
        super().__init__()
        self.lstm = lstm
        self.state_size = state_size
        self.action_size = action_size
        if lstm:
            self.encoder = encoder

            self.net = nn.Sequential(
                nn.Linear(hidden_size + action_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(state_size + action_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size),
            )

    def forward(self, state, action):
        if self.lstm:
            state = self.encoder(state)
            return self.net(torch.cat([state, action], dim=1))
        else:
            return self.net(torch.cat([state.view(-1, self.state_size), action.view(-1, self.action_size)], dim=1))


class DDPG:
    def __init__(self, gamma, tau, hidden_size, state_size, action_size, device="cpu", batch_size=32, count_last_states=1, path_save_anim=None, path_save_model=None, actor=None):
        self.gamma = gamma
        self.tau = tau
        self.action_size = action_size
        self.device = device
        self.batch_size = batch_size
        self.count_last_states = count_last_states
        self.lstm = count_last_states > 1
        self.path_save_anim = path_save_anim
        self.path_save_model = path_save_model

        self.encoder = LSTMEncoder(state_dim=state_size, hidden_size=hidden_size)
        # Define the actor
        if actor is None:
            self.actor = ActorNetwork(encoder=self.encoder, input_size=state_size, hidden_size=hidden_size, output_size=action_size, lstm=self.lstm).to(self.device)
            self.actor_target = ActorNetwork(encoder=self.encoder, input_size=state_size, hidden_size=hidden_size, output_size=action_size, lstm=self.lstm).to(self.device)
        else:
            self.actor = actor.to(self.device)
            self.actor_target = deepcopy(actor).to(self.device)

        # Define the critic
        self.critic = CriticNetwork(encoder=self.encoder, state_size=state_size, action_size=action_size, hidden_size=hidden_size, output_size=1, lstm=self.lstm).to(device)
        self.critic_target = CriticNetwork(encoder=self.encoder, state_size=state_size, action_size=action_size, hidden_size=hidden_size, output_size=1, lstm=self.lstm).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)  # optimizer for the actor network
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)  # optimizer for the critic network

        self.replay_buffer = ReplayBuffer(int(1e5), lstm=self.lstm)
        self.noise = GaussianNoise(action_dim=action_size)

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def calc_action(self, state, action_noise=None):
        x = state.to(self.device)

        with torch.no_grad():
            action = self.actor(x)

        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(self.device)
            action += noise

        # Clip the output according to the action space of the env
        action = action.clamp(-1, 1)

        return action

    def load_actor(self, path):
        self.actor = torch.load(path, weights_only=False)

    def update_params(self):
        """
        Updates the parameters/networks of the agent according to the given batch.
        This means we ...
            1. Compute the targets
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update

        Arguments:
            batch:  Batch to perform the training of the parameters
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0

            # Берем batch из буфера
        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)

        # # Get tensors from the batch
        state_batch = states.to(self.device)
        action_batch = actions.to(self.device)
        reward_batch = rewards.to(self.device)
        next_state_batch = next_states.to(self.device)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        expected_values = reward_batch + self.gamma * next_state_action_values

        expected_values = torch.clamp(expected_values, -40, 40)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = nn.functional.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic_target.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.actor_target.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        return value_loss.item(), policy_loss.item()

    def train(self, env, epochs=10, count_steps=200):
        for epoch in range(epochs):
            dog, target = env.reset()
            epoch_critic_loss = 0
            epoch_actor_loss = 0
            num_updates = 0
            fitness = 0
            for i in range(count_steps):
                state = env.get_state(dog, target, self.count_last_states, save=True)
                state = torch.FloatTensor(state).to(self.device)

                if epoch < epochs // 2:
                    action = self.calc_action(state.unsqueeze(0), self.noise)
                else:
                    action = self.calc_action(state.unsqueeze(0))

                if self.device == "cuda":
                    action = action.cpu()
                action = action.squeeze(0)

                if not self.lstm:
                    action = action.squeeze(0)

                dog, target, reward, done = env.step(dog, target, action[0], action[1], i, count_steps)
                next_state = env.get_state(dog, target, save=False)
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                fitness += reward

                self.replay_buffer.push(state, action.unsqueeze(0), reward, next_state)

                critic_loss, policy_loss = self.update_params()
                if critic_loss != 0:
                    epoch_critic_loss += critic_loss
                    epoch_actor_loss += policy_loss
                    num_updates += 1

                if done:
                    break


            if num_updates == 0:
                print(f"value_loss: -, policy_loss: -, fitness: {fitness}")
            else:
                print(f"value_loss: {epoch_critic_loss / num_updates}, policy_loss: {epoch_actor_loss / num_updates}, fitness: {fitness}")
            if epoch % 100 == 0:
                if self.path_save_anim is None:
                    path = "anim_generation"
                else:
                    path = self.path_save_anim
                Visualizer.animation(Evolution.Individual(self.actor, device="cuda"), env=env, device="cuda",
                                     render=False, save_path=path + "/DDPG_epoch_" + str(epoch))

                if self.path_save_model is None:
                    path = "pretraining_model"
                else:
                    path = self.path_save_model
                torch.save(self.actor_target, f"./{path}/DDPG_" + str(epoch) + f"_{fitness:.4f}" + ".pth")
