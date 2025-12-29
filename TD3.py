import torch
import torch.nn as nn
import Evolution
import Visualizer
from collections import deque, namedtuple
import random

Experience = namedtuple('Experience',
                       ('state', 'action', 'reward', 'next_state', 'done'))


class GaussianNoise:
    def __init__(self, action_dim, sigma=0.1, sigma_min=0.01, sigma_decay=0.999):
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

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        if self.lstm:
            states = torch.nn.utils.rnn.pad_sequence(
                [e.state for e in experiences],
                batch_first=True,
                padding_value=0.0
            )
        else:
            states = torch.cat([e.state for e in experiences])
        actions = torch.cat([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.cat([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])

        return states, actions, rewards, next_states, dones

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


class TwinCriticNetwork(nn.Module):
    def __init__(self, encoder, state_size=42, action_size=2, hidden_size=64, output_size=1, lstm=False):
        super().__init__()
        self.lstm = lstm
        self.state_size = state_size
        self.action_size = action_size
        if lstm:
            self.encoder = encoder

            self.q1 = nn.Sequential(
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
            self.q2 = nn.Sequential(
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
            self.q1 = nn.Sequential(
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
            self.q2 = nn.Sequential(
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
        q1, q2 = self.get_target_q(state, action)
        return torch.min(q1, q2)

    def get_target_q(self, state, action):
        if self.lstm:
            state = self.encoder(state)
            q1 = self.q1(torch.cat([state, action], dim=1))
            q2 = self.q2(torch.cat([state, action], dim=1))
        else:
            q1 = self.q1(torch.cat([state.view(-1, self.state_size), action.view(-1, self.action_size)], dim=1))
            q2 = self.q2(torch.cat([state.view(-1, self.state_size), action.view(-1, self.action_size)], dim=1))
        return q1, q2

    def Q1(self, state, action):
        q1, q2 = self.get_target_q(state, action)
        return torch.min(q1, q2)


class TD3:
    def __init__(self, gamma, tau, hidden_size, state_size, action_size, device="cpu", batch_size=32, count_last_states=1, policy_update_freq=2, policy_noise=0.3, noise_clip=1, path_actor=None, path_save_anim=None, path_save_model=None):
        self.gamma = gamma
        self.tau = tau
        self.action_size = action_size
        self.device = device
        self.batch_size = batch_size
        self.count_last_states = count_last_states
        self.lstm = count_last_states > 1
        self.policy_update_freq = policy_update_freq
        self.total_steps = 0
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.path_save_anim = path_save_anim
        self.path_save_model = path_save_model

        self.encoder_actor = LSTMEncoder(state_dim=state_size, hidden_size=hidden_size)
        self.encoder_critic = LSTMEncoder(state_dim=state_size, hidden_size=hidden_size)
        # Define the actor
        if path_actor is None:
            self.actor = ActorNetwork(encoder=self.encoder_actor, input_size=state_size, hidden_size=hidden_size, output_size=action_size, lstm=self.lstm).to(self.device)
        else:
            self.load_actor(path_actor)
        self.actor_target = ActorNetwork(encoder=LSTMEncoder(state_dim=state_size, hidden_size=hidden_size), input_size=state_size, hidden_size=hidden_size, output_size=action_size, lstm=self.lstm).to(self.device)

        # Define the critic
        self.critic = TwinCriticNetwork(encoder=self.encoder_critic, state_size=state_size, action_size=action_size, hidden_size=hidden_size, output_size=1, lstm=self.lstm).to(device)
        self.critic_target = TwinCriticNetwork(encoder=LSTMEncoder(state_dim=state_size, hidden_size=hidden_size), state_size=state_size, action_size=action_size, hidden_size=hidden_size, output_size=1, lstm=self.lstm).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)  # optimizer for the actor network
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)  # optimizer for the critic network

        self.replay_buffer = ReplayBuffer(int(1e4), lstm=self.lstm)
        self.noise = GaussianNoise(action_dim=action_size)

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def load_actor(self, path):
        self.actor = torch.load(path, weights_only=False)

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

    def update_params(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0

            # Берем batch из буфера
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # # Get tensors from the batch
        state_batch = states.to(self.device)
        action_batch = actions.to(self.device)
        reward_batch = rewards.to(self.device)
        # reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)
        next_state_batch = next_states.to(self.device)
        done_batch = dones.to(self.device)

        noise = torch.randn_like(actions) * self.policy_noise
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip).to(self.device)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch) + noise
        next_action_batch = torch.clamp(next_action_batch, -1, 1)
        target_q = self.critic_target(next_state_batch.detach(), next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        expected_values = reward_batch + self.gamma * target_q * (1 - done_batch).view(-1, 1)

        # Update the critic network
        with torch.autograd.set_detect_anomaly(False):
            self.critic_optimizer.zero_grad()
            current_q1, current_q2 = self.critic.get_target_q(state_batch, action_batch)
            critic_loss = nn.functional.mse_loss(current_q1, expected_values.detach()) + nn.functional.mse_loss(current_q2, expected_values.detach())
            critic_loss.backward(retain_graph=False)
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

            # Update the actor network
            self.actor_optimizer.zero_grad()
            actor_loss = -self.critic.Q1(state_batch, self.actor(state_batch)).mean()

            if self.total_steps % self.policy_update_freq == 0:
                actor_loss.backward(retain_graph=False)
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()
                soft_update(self.actor_target, self.actor, self.tau)
                soft_update(self.critic_target, self.critic, self.tau)

        self.total_steps += 1
        return critic_loss.item(), actor_loss.item()

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

                if epoch < epochs // 5:
                    action = self.calc_action(state.unsqueeze(0), self.noise)
                else:
                    action = self.calc_action(state.unsqueeze(0))
                # action = self.calc_action(state.unsqueeze(0))

                if self.device == "cuda":
                    action = action.cpu()
                action = action.squeeze(0)

                if not self.lstm:
                    action = action.squeeze(0)

                dog, target, reward, done = env.step(dog, target, action[0], action[1], i, count_steps)
                next_state = env.get_state(dog, target, save=False)
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                fitness += reward

                self.replay_buffer.push(state, action.unsqueeze(0), reward, next_state, done)

                critic_loss, policy_loss = self.update_params()
                if critic_loss != 0:
                    epoch_critic_loss += critic_loss
                    epoch_actor_loss += policy_loss
                    num_updates += 1

                if done:
                    break

            torch.cuda.empty_cache() if self.device == "cuda" else None

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
                                     render=False, save_path=path+"/DDPG_epoch_"+str(epoch))

                if self.path_save_model is None:
                    path = "pretraining_model"
                else:
                    path = self.path_save_model
                torch.save(self.actor_target, f"./{path}/DDPG_" + str(epoch) + f"_{fitness:.4f}" + ".pth")
