import torch
import torch.nn as nn
import Evolution
import Visualizer
from collections import deque, namedtuple
import random
import numpy as np

Experience = namedtuple('Experience',
                       ('state', 'action', 'reward', 'done', 'log_probs', 'value'))


class ActorNetwork(Evolution.EvolutionaryNetwork):
    def __init__(self, encoder=None, input_size=8, hidden_size=64, output_size=2, lstm=False):
        super().__init__()
        self.lstm = lstm
        self.output_size = output_size
        if lstm:
            self.encoder = encoder

            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, output_size),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, output_size),
            )

    def forward(self, x):
        if self.lstm:
            x = self.encoder(x)
            return self.net(x)
        else:
            return self.net(x).view(-1, self.output_size)

    def get_action(self, x):
        output = self.net(x)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -3, 0)
        std = torch.exp(log_std)
        std = torch.clamp(std, min=1e-4, max=2.0)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_tanh = torch.tanh(action)
        return action_tanh


class CriticNetwork(nn.Module):
    def __init__(self, encoder=None, state_size=42, hidden_size=64, output_size=1, lstm=False):
        super().__init__()
        self.lstm = lstm
        self.state_size = state_size
        if lstm:
            self.encoder = encoder
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, state):
        if self.lstm:
            state = self.encoder(state)
            return self.net(state)
        else:
            return self.net(state.view(-1, self.state_size))


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


class PPO:
    def __init__(self, gamma, gae_lambda, hidden_size, state_size, action_size, device="cpu", actor=None, batch_size=64, count_last_states=1, clip_epsilon=0.1, path_actor=None, trajectory_length=512, count_retrain=3, path_save_anim=None, path_save_model=None):
        self.gamma = gamma
        self.action_size = action_size
        self.device = device
        self.batch_size = batch_size
        self.count_last_states = count_last_states
        self.lstm = count_last_states > 1
        self.total_steps = 0
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.path_save_anim = path_save_anim
        self.path_save_model = path_save_model
        self.initial_exploration_steps = 0
        self.noise_sigma = 0.3
        self.noise_decay = 0.999
        self.entropy_coef = 0.1
        self.min_entropy_coef = 0.05

        self.encoder_actor = LSTMEncoder(state_dim=state_size, hidden_size=hidden_size)
        self.encoder_critic = LSTMEncoder(state_dim=state_size, hidden_size=hidden_size)
        # Define the actor
        if actor is not None:
            self.actor = actor
        else:
            if path_actor is None:
                self.actor = ActorNetwork(encoder=self.encoder_actor, input_size=state_size, hidden_size=hidden_size, output_size=self.action_size, lstm=self.lstm).to(device)
            else:
                self.load_actor(path_actor)

        # Define the critic
        self.critic = CriticNetwork(encoder=self.encoder_critic, state_size=state_size, hidden_size=hidden_size, output_size=1, lstm=self.lstm).to(device)

        self.count_retrain = count_retrain
        self.log_std = nn.Parameter(torch.full((1, action_size), -2.0, device=self.device))
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': 3e-4},
            {'params': self.critic.parameters(), 'lr': 3e-4},
            {'params': [self.log_std], 'lr': 3e-4}
        ])

        self.value_clip = True
        self.value_clip_epsilon = 0.2

        self.trajectory_buffer = []
        self.trajectory_length = trajectory_length

        self._initialize_critic_small()

    def _initialize_critic_small(self):
        """Инициализация критика с маленькими весами"""
        for layer in self.critic.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0.0)

    def get_action_distribution(self, state):
        """Получение нормального распределения с обучаемым std"""
        log_std_clamped = torch.clamp(self.log_std, -5, 2)
        mean = self.actor(state).to(self.device)

        if self.total_steps < self.initial_exploration_steps:
            exploration_factor = 1.0
            extra_std = self.noise_sigma * exploration_factor
            std = torch.exp(log_std_clamped) + extra_std
        else:
            std = torch.exp(log_std_clamped)

        std = torch.clamp(std, 0.1, 2)
        std = std.expand_as(mean).to(self.device)
        return torch.distributions.Normal(mean, std)

    def get_log_prob(self, dist, action):
        eps = 1e-2
        action_clamped = torch.clamp(action, -1.0 + eps, 1.0 - eps)
        action_raw = torch.atanh(action_clamped)
        log_prob_raw = dist.log_prob(action_raw)
        log_det_jacobian = torch.log(1 - action_clamped.pow(2) + eps)
        log_prob = (log_prob_raw - log_det_jacobian).sum(dim=-1)
        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
            log_prob = log_prob_raw.sum(dim=-1)
            log_prob = torch.clamp(log_prob, min=-100, max=100)
        return log_prob

    def compute_gae(self, rewards, values, dones, next_value):
        batch_size = len(rewards)
        advantages = torch.zeros(batch_size).to(self.device)
        returns = torch.zeros(batch_size).to(self.device)

        gae = 0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_values = next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_values = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1]

            # TD-error
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            # GAE
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            if dones[t]:
                gae = 0

            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def load_actor(self, path):
        self.actor = torch.load(path, weights_only=False)

    def get_params(self):
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        for i in self.trajectory_buffer:
            states.append(i['state'])
            actions.append(i['action'])
            rewards.append(i['reward'])
            dones.append(i['done'])
            log_probs.append(i['log_prob'])
            values.append(i['value'])

        return torch.stack(states).detach().to(self.device), torch.stack(actions).detach().to(self.device),\
               torch.stack(rewards).detach().to(self.device), torch.stack(dones).detach().to(self.device),\
               torch.stack(log_probs).detach().to(self.device), torch.stack(values).detach().to(self.device)

    def update_params(self):
        # Получаем данные из буфера
        states, actions, rewards, dones, old_log_probs, old_values = self.get_params()

        # Вычисляем advantages и returns с помощью GAE
        # Для последнего состояния нужен next_value
        with torch.no_grad():
            if dones[-1]:
                next_value = torch.tensor(0.0, device=self.device)
            else:
                # Получаем value для следующего состояния (последнее состояние в траектории)
                next_state = states[-1].unsqueeze(0) if states[-1].dim() == 1 else states[-1:].unsqueeze(0)
                next_value = self.critic(next_state).squeeze()

        # Вычисляем advantages и returns
        advantages, returns = self.compute_gae(rewards, old_values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Обновляем несколько раз (PPO epochs)
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0
        indices = torch.randperm(self.trajectory_length, device=self.device)

        for epoch in range(self.count_retrain):
            for start_idx in range(0, self.trajectory_length, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.trajectory_length)
                batch_indices = indices[start_idx:end_idx]

                # Берем мини-батч
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                # batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                batch_returns = returns[batch_indices]

                # Обнуляем градиенты для каждого мини-батча
                self.optimizer.zero_grad()

                # 1. Получаем новые предсказания от текущей политики
                dist = self.get_action_distribution(batch_states)
                new_log_probs = self.get_log_prob(dist, batch_actions)

                # 2. Добавляем энтропию для регуляризации
                entropy = dist.entropy().mean()

                # 3. Вычисляем ratio и PPO loss
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.entropy_coef = max(self.min_entropy_coef,
                                        self.entropy_coef * self.noise_decay)

                # 4. Обновляем критика
                values_pred = self.critic(batch_states).squeeze()
                if self.value_clip:
                    values_clipped = torch.flatten(old_values[batch_indices]) + \
                                     torch.clamp(torch.flatten(values_pred) - torch.flatten(old_values[batch_indices]),
                                                 -self.value_clip_epsilon,
                                                 self.value_clip_epsilon)
                    loss_vf1 = nn.functional.mse_loss(values_pred, batch_returns)
                    loss_vf2 = nn.functional.mse_loss(values_clipped, batch_returns)
                    critic_loss = 0.5 * torch.max(loss_vf1, loss_vf2)
                else:
                    critic_loss = 0.5 * nn.functional.mse_loss(values_pred, batch_returns)

                # 5. Общий loss
                loss = actor_loss + 0.5 * critic_loss

                # 6. Backward и шаг оптимизации
                loss.backward()

                # Обрезаем градиенты
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_([self.log_std], 0.1)

                # Шаг оптимизации для каждого мини-батча
                self.optimizer.step()

                # Накопление статистики
                batch_size = end_idx - start_idx
                total_actor_loss += actor_loss.item() * batch_size
                total_critic_loss += critic_loss.item() * batch_size
                total_loss += loss.item() * batch_size

                # Очистка памяти после каждого мини-батча
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                # Перемешиваем индексы для следующей эпохи
            indices = torch.randperm(self.trajectory_length, device=self.device)

        # Обновляем счетчик шагов
        self.total_steps += 1
        total_actor_loss /= (self.trajectory_length * self.count_retrain)
        total_critic_loss /= (self.trajectory_length * self.count_retrain)
        total_loss /= (self.trajectory_length * self.count_retrain)

        # Очищаем буфер после обновления
        self.trajectory_buffer.clear()

        # Возвращаем средние по эпохам
        return (total_critic_loss / self.count_retrain,
                total_actor_loss / self.count_retrain,
                total_loss / self.count_retrain)

    def train(self, env, epochs=10, count_steps=200):
        self.initial_exploration_steps = epochs * count_steps // 5
        for epoch in range(epochs):
            dog, target = env.reset()
            fitness = 0
            self.trajectory_buffer.clear()

            if self.device == "cuda":
                torch.cuda.empty_cache()
            entropy = 0

            for i in range(self.trajectory_length):  # Собираем N шагов
                state = env.get_state(dog, target, self.count_last_states, save=True)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # 1. Получаем распределение С ОБУЧАЕМЫМ STD
                dist = self.get_action_distribution(state_tensor)
                entropy = dist.entropy().mean()

                # 2. Сэмплируем действие
                action_raw = dist.rsample()
                action = torch.tanh(action_raw)  # Squashing

                # 3. ВЫЧИСЛЯЕМ log_prob ПРАВИЛЬНО
                log_prob = self.get_log_prob(dist, action)

                # 4. Получаем value
                with torch.no_grad():
                    value = self.critic(state_tensor).detach()

                # 5. Применяем в среде
                action_np = action.detach().cpu().numpy()[0]
                dog, target, reward, done = env.step(
                    dog, target, action_np[0], action_np[1], i, count_steps
                )
                fitness += reward

                self.trajectory_buffer.append({
                    'state': state_tensor.cpu(),
                    'action': action.cpu(),
                    'reward': torch.FloatTensor([reward]).cpu(),
                    'done': torch.FloatTensor([done]).cpu(),
                    'log_prob': log_prob.unsqueeze(0).cpu(),
                    'value': value.cpu()
                })

                if done:
                    dog, target = env.reset()

            critic_loss, actor_loss, loss = self.update_params()
            fitness /= self.trajectory_length

            print(f"Epoch {epoch}: Critic={critic_loss:.4f}, Actor={actor_loss:.4f}, "
                  f"Loss={loss}, Fitness={fitness:.2f}, Entropy={entropy.item():.4f}, "
                  f"Log_std={self.log_std.mean().item():.4f}")
            if epoch % 100 == 0:
                if self.path_save_anim is None:
                    path = "anim_generation"
                else:
                    path = self.path_save_anim
                Visualizer.animation(Evolution.Individual(self.actor, device="cuda"), env=env, device="cuda",
                                     render=False, save_path=path + "/PPO_epoch_" + str(epoch))

                if self.path_save_model is None:
                    path = "pretraining_model"
                else:
                    path = self.path_save_model
                torch.save(self.actor, f"./{path}/PPO_" + str(epoch) + f"_{fitness:.4f}" + ".pth")


class SimplePPO:
    def __init__(self, gamma, gae_lambda, hidden_size, state_size, action_size, device="cpu", actor=None, batch_size=64,
                 clip_epsilon=0.2, trajectory_length=512, count_retrain=4, path_save_anim=None, path_save_model=None):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.action_size = action_size
        self.device = device
        self.batch_size = batch_size
        self.clip_epsilon = clip_epsilon
        self.trajectory_length = trajectory_length
        self.count_retrain = count_retrain
        self.path_save_anim = path_save_anim
        self.path_save_model = path_save_model

        if actor is not None:
            self.actor = actor
        else:
            self.actor = ActorNetwork(input_size=state_size, hidden_size=hidden_size, output_size=self.action_size * 2).to(device)

        self.reinitialize_network()

        # Define critic network
        self.critic = CriticNetwork(state_size=state_size, hidden_size=hidden_size, output_size=1).to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4, eps=1e-8)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        # Trajectory buffer
        self.trajectory_buffer = []

    def get_action_distribution(self, state):
        """Get normal distribution for actions"""
        output = self.actor(state)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -3, 0)
        std = torch.exp(log_std)
        std = torch.clamp(std, min=1e-4, max=2.0)
        return torch.distributions.Normal(mean, std)

    def get_log_prob(self, dist, action_raw, action_tanh):
        """Корректный расчет log_prob при использовании tanh"""
        # log_prob = log_prob_raw - log(1 - tanh^2(x) + eps)
        log_prob_raw = dist.log_prob(action_raw).sum(dim=-1)
        log_det_jacobian = 2.0 * (np.log(2) - action_raw - nn.functional.softplus(-2.0 * action_raw))
        log_det_jacobian = log_det_jacobian.sum(dim=-1)

        log_prob = log_prob_raw - log_det_jacobian
        return torch.clamp(log_prob, -50, 50)

    def reinitialize_network(self):
        for layer in self.actor.modules():
            if isinstance(layer, nn.Linear):
                # Правильная инициализация для Tanh/ReLU
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def get_action(self, state):
        """Sample action from policy"""
        with torch.no_grad():
            state_tensor = state.to(self.device)
            dist = self.get_action_distribution(state_tensor)
            action = dist.sample()
            action_tanh = torch.tanh(action)
            log_prob = self.get_log_prob(dist, action, action_tanh)
            value = self.critic(state_tensor).squeeze()
            return action.cpu(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros(len(rewards)).to(self.device)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]

            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

            if dones[t]:
                gae = 0

        # Calculate returns
        returns = advantages + values
        # Normalize advantages

        advantages_mean = advantages.mean()
        advantages_std = advantages.std()
        if torch.isnan(advantages_std) or advantages_std < 1e-8:
            print("⚠️ Слишком маленький или NaN std advantages!")
            advantages = torch.zeros_like(advantages)
        else:
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def collect_trajectory(self, env, max_steps=1000):
        """Collect trajectory data"""
        self.trajectory_buffer.clear()

        dog, target = env.reset()
        state = env.get_state(dog, target, 1, save=True)
        episode_reward = 0

        for step in range(self.trajectory_length):
            # Get action from policy
            action, log_prob, value = self.get_action(state)

            # Take action in environment
            dog, target, reward, done = env.step(
                dog, target, action[0, 0], action[0, 1], step, max_steps
            )
            episode_reward += reward

            # Store transition
            self.trajectory_buffer.append({
                    'state': state.cpu(),
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'log_prob': log_prob,
                    'value': value
                })

            state = env.get_state(dog, target, 1, save=True)

            if done or step >= max_steps:
                dog, target = env.reset()
                state = env.get_state(dog, target, 1, save=True)
                episode_reward = 0

        return episode_reward / self.trajectory_length

    def update(self):
        """Update policy using PPO"""
        if len(self.trajectory_buffer) < self.batch_size:
            return 0, 0, 0

        # Convert buffer to tensors
        states = torch.cat([t['state'] for t in self.trajectory_buffer], dim=0).to(self.device)
        actions = torch.cat([t['action'] for t in self.trajectory_buffer], dim=0).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in self.trajectory_buffer]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in self.trajectory_buffer]).to(self.device)
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in self.trajectory_buffer]).to(self.device)
        old_values = torch.FloatTensor([t['value'] for t in self.trajectory_buffer]).to(self.device)

        # Get next value for GAE
        with torch.no_grad():
            if dones[-1]:
                next_value = torch.tensor(0.0, device=self.device)
            else:
                next_state = states[-1:].to(self.device)
                next_value = self.critic(next_state).squeeze()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values, dones, next_value)

        # PPO update for multiple epochs
        total_actor_loss = 0
        total_critic_loss = 0

        for epoch in range(self.count_retrain):
            # Shuffle indices
            indices = torch.randperm(len(self.trajectory_buffer), device=self.device)

            # Mini-batch update
            for start_idx in range(0, len(self.trajectory_buffer), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.trajectory_buffer))
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # ----- Update Actor -----
                dist = self.get_action_distribution(batch_states)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                entropy = dist.entropy().mean()
                entropy_coef = 0.01  # Начните с этого

                # PPO ratio and clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

                # ----- Update Critic -----
                values_pred = self.critic(batch_states).squeeze()

                returns_mean = batch_returns.mean()
                returns_std = batch_returns.std()

                if returns_std > 1e-6:
                    returns_normalized = (batch_returns - returns_mean) / (returns_std + 1e-8)
                    returns_normalized = torch.clamp(returns_normalized, -5, 5)
                else:
                    returns_normalized = torch.zeros_like(batch_returns)

                # Нормализация predictions
                values_pred_normalized = (values_pred - values_pred.mean()) / (values_pred.std() + 1e-8)

                critic_loss = nn.functional.mse_loss(values_pred_normalized, returns_normalized)

                # ----- Backpropagation -----
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1)
                torch.nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=0.1)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.1)
                torch.nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=0.1)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        # Clear buffer
        self.trajectory_buffer.clear()

        # Average losses
        avg_actor_loss = total_actor_loss / self.count_retrain
        avg_critic_loss = total_critic_loss / self.count_retrain

        return avg_actor_loss, avg_critic_loss, avg_actor_loss + avg_critic_loss

    def train(self, env, epochs=1000, eval_interval=100):
        """Main training loop"""
        for epoch in range(epochs):
            # Collect trajectory
            avg_reward = self.collect_trajectory(env)

            # Update policy
            actor_loss, critic_loss, total_loss = self.update()

            if torch.isnan(torch.tensor(actor_loss)) or torch.isinf(torch.tensor(actor_loss)):
                print(f"ВНИМАНИЕ: NaN/Inf в потерях на эпохе {epoch}")

            # Logging
            print(f"Epoch {epoch}: "
                    f"Actor Loss: {actor_loss:.4f}, "
                    f"Critic Loss: {critic_loss:.4f}, "
                    f"Total Loss: {total_loss:.4f}, "
                    f"Avg Reward: {avg_reward:.2f}")

            # Evaluation
            if epoch % eval_interval == 0:
                self.evaluate(env, num_episodes=5)

                if self.path_save_anim is None:
                    path = "anim_generation"
                else:
                    path = self.path_save_anim
                Visualizer.animation(Evolution.Individual(self.actor, device="cuda"), env=env, device="cuda",
                                     render=False, save_path=path + "/PPO_epoch_" + str(epoch), get_action=True)

                if self.path_save_model is None:
                    path = "pretraining_model"
                else:
                    path = self.path_save_model
                torch.save(self.actor, f"./{path}/PPO_" + str(epoch) + f"_{avg_reward:.4f}" + ".pth")

    def evaluate(self, env, num_episodes=10):
        """Evaluate current policy"""
        total_rewards = []

        for episode in range(num_episodes):
            dog, target = env.reset()
            state = env.get_state(dog, target, 1, save=True)
            episode_reward = 0
            done = False

            while not done:
                action, _, _ = self.get_action(state)
                dog, target, reward, done = env.step(
                    dog, target, action[0, 0], action[0, 1], 0, 300
                )
                episode_reward += reward

            total_rewards.append(episode_reward)

        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Evaluation: Avg Reward over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward

    def save_model(self, path):
        """Save model weights"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

    def load_model(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])