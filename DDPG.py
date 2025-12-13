import torch
import torch.nn as nn
import Evolution
import Visualizer
from collections import deque, namedtuple
import random

Experience = namedtuple('Experience',
                       ('state', 'action', 'reward', 'next_state'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append(Experience(state, action, reward, next_state))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

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


class ActorNetwork(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class CriticNetwork(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, state, action):
        return self.net(torch.hstack([state, action]))


class DDPG:
    def __init__(self, gamma, tau, hidden_size, state_size, action_size, device="cpu", batch_size=32):
        self.gamma = gamma
        self.tau = tau
        self.action_size = action_size
        self.device = device
        self.batch_size = batch_size

        # Define the actor
        # self.actor = ActorNetwork(input_size=state_size, hidden_size=hidden_size, output_size=action_size).to(self.device)
        self.actor = Evolution.EvolutionaryNetwork(input_size=state_size, hidden_size=hidden_size, output_size=action_size).to(self.device)
        self.actor_target = Evolution.EvolutionaryNetwork(input_size=state_size, hidden_size=hidden_size, output_size=action_size).to(self.device)

        # Define the critic
        self.critic = CriticNetwork(input_size=state_size+action_size, hidden_size=hidden_size, output_size=1).to(device)
        self.critic_target = CriticNetwork(input_size=state_size+action_size, hidden_size=hidden_size, output_size=1).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)  # optimizer for the actor network
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)  # optimizer for the critic network
        self.replay_buffer = ReplayBuffer(10000)

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
        # state_batch = torch.cat(states).to(self.device)
        # action_batch = torch.cat(actions).view(-1, self.action_size).to(self.device)
        # reward_batch = rewards.to(self.device)
        # next_state_batch = torch.cat(next_states).to(self.device)
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

        # expected_value = torch.clamp(expected_value, min_value, max_value)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = nn.functional.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
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
                state = env.get_state(dog, target)
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.calc_action(state)
                if self.device == "cuda":
                    action = action.cpu()
                action = action.squeeze(0)
                dog, target, reward = env.step(dog, target, action[0], action[1])
                next_state = env.get_state(dog, target)
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                fitness += reward

                self.replay_buffer.push(state, action.unsqueeze(0), reward, next_state)

                critic_loss, policy_loss = self.update_params()
                if critic_loss != 0:  # Если было обновление
                    epoch_critic_loss += critic_loss
                    epoch_actor_loss += policy_loss
                    num_updates += 1

            print(f"value_loss: {epoch_critic_loss / num_updates}, policy_loss: {epoch_actor_loss / num_updates}, avg_fitness: {fitness}")
            if epoch % 100 == 0:
                Visualizer.animation(Evolution.Individual(self.actor, device="cuda"), env=env, device="cuda",
                                     render=False, save_path="anim_generation"+"/DDPG_epoch_"+str(epoch))
                torch.save(self.actor_target, "./pretraining_model/DDPG_" + str(epoch) + f"_{fitness:.4f}" + ".pth")
