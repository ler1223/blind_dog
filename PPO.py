import torch
import torch.nn as nn
import Evolution
import Visualizer
from collections import deque, namedtuple
import random

Experience = namedtuple('Experience',
                       ('state', 'action', 'reward', 'done', 'log_probs', 'value'))


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


class CriticNetwork(nn.Module):
    def __init__(self, encoder, state_size=42, hidden_size=64, output_size=1, lstm=False):
        super().__init__()
        self.lstm = lstm
        self.state_size = state_size
        if lstm:
            self.encoder = encoder
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
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

    def forward(self, state):
        if self.lstm:
            state = self.encoder(state)
            return self.net(state)
        else:
            return self.net(state.view(-1, self.state_size))


class PPO:
    def __init__(self, gamma, gae_lambda, hidden_size, state_size, action_size, device="cpu", batch_size=32, count_last_states=1, policy_update_freq=2, clip_epsilon=0.1, path_actor=None, count_retrain=10, path_save_anim=None, path_save_model=None):
        self.gamma = gamma
        self.action_size = action_size
        self.device = device
        self.batch_size = batch_size
        self.count_last_states = count_last_states
        self.lstm = count_last_states > 1
        self.policy_update_freq = policy_update_freq
        self.total_steps = 0
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.path_save_anim = path_save_anim
        self.path_save_model = path_save_model

        self.encoder_actor = LSTMEncoder(state_dim=state_size, hidden_size=hidden_size)
        self.encoder_critic = LSTMEncoder(state_dim=state_size, hidden_size=hidden_size)
        # Define the actor
        if path_actor is None:
            self.actor = ActorNetwork(encoder=self.encoder_actor, input_size=state_size, hidden_size=hidden_size, output_size=action_size, lstm=self.lstm).to(self.device)
        else:
            self.load_actor(path_actor)

        # Define the critic
        self.critic = CriticNetwork(encoder=self.encoder_critic, state_size=state_size, hidden_size=hidden_size, output_size=1, lstm=self.lstm).to(device)

        self.count_retrain = count_retrain
        self.log_std = nn.Parameter(torch.zeros(1, action_size).to(self.device))
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()},
            {'params': [self.log_std], 'lr': 1e-4}
        ], lr=1e-4)

        self.trajectory_buffer = []
        self.trajectory_length = 2048
        self.noise = GaussianNoise(action_dim=action_size)

    def get_action_distribution(self, state):
        """Получение нормального распределения с обучаемым std"""
        mean = self.actor(state).to(self.device)
        std = torch.exp(self.log_std).expand_as(mean).to(self.device)  # Обучаемое std
        return torch.distributions.Normal(mean, std)

    def get_log_prob(self, dist, action):
        # action уже прошел через tanh
        # Преобразуем обратно через atanh
        action_raw = torch.atanh(torch.clamp(action, -0.999, 0.999))
        log_prob = dist.log_prob(action_raw)
        # Учет Jacobian tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

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

        # ВАЖНО: отключаем градиенты для old_log_probs и old_values
        old_log_probs = old_log_probs.detach()
        old_values = old_values.detach()

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
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

                # 4. Обновляем критика
                values_pred = self.critic(batch_states).squeeze()
                critic_loss = torch.nn.functional.mse_loss(values_pred, batch_returns)

                # 5. Общий loss
                loss = actor_loss + 0.5 * critic_loss

                # 6. Backward и шаг оптимизации
                loss.backward()

                # Обрезаем градиенты
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_([self.log_std], 1.0)

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
        for epoch in range(epochs):
            dog, target = env.reset()
            fitness = 0
            self.trajectory_buffer.clear()

            if self.device == "cuda":
                torch.cuda.empty_cache()

            for i in range(self.trajectory_length):  # Собираем N шагов
                state = env.get_state(dog, target, self.count_last_states, save=True)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # 1. Получаем распределение С ОБУЧАЕМЫМ STD
                dist = self.get_action_distribution(state_tensor)

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

            print(f"Epoch {epoch}: Critic={critic_loss:.4f}, Actor={actor_loss:.4f}, Loss={loss}, Fitness={fitness:.2f}")
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
                torch.save(self.actor, f"./{path}/DDPG_" + str(epoch) + f"_{fitness:.4f}" + ".pth")
