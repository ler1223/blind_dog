import torch
import torch.nn as nn
import Evolution
import Visualizer
from collections import deque, namedtuple
import random


class ActorNetwork(Evolution.EvolutionaryNetwork):
    def __init__(self, encoder, input_size=8, hidden_size=64, output_size=2, lstm=False):
        super().__init__()
        self.output_size = output_size
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
        return self.net(x).view(-1, self.output_size)


class SimpleActionArbiter:
    def __init__(self):
        pass

    def combine(self, actions_dict, weights):
        combined = torch.zeros_like(next(iter(actions_dict.values())))
        total_weight = 0.0

        for skill_name, action in actions_dict.items():
            weight = weights[skill_name]
            combined += action * weight
            total_weight += weight

        if total_weight > 0:
            combined = combined / total_weight

        combined = torch.tanh(combined)
        return combined

    def combine_batch(self, actions_dict, weights_dict):
        """
        actions_dict: dict {skill_name: tensor[batch_size, action_dim]}
        weights_dict: dict {skill_name: tensor[batch_size]}
        Возвращает: tensor[batch_size, action_dim]
        """
        # Инициализируем нулями
        batch_size = next(iter(actions_dict.values())).shape[0]
        action_dim = next(iter(actions_dict.values())).shape[1]
        device = next(iter(actions_dict.values())).device

        combined = torch.zeros(batch_size, action_dim, device=device)
        total_weight = torch.zeros(batch_size, 1, device=device)

        # Суммируем взвешенные действия
        for skill_name in actions_dict:
            weight = weights_dict[skill_name].unsqueeze(1)  # [batch_size, 1]
            action = actions_dict[skill_name]  # [batch_size, action_dim]

            combined += weight * action
            total_weight += weight

        # Нормализуем (избегаем деления на 0)
        combined = combined / (total_weight + 1e-8)

        return combined


class SimpleHierarchicalActor(nn.Module):

    def __init__(self, pretrained_skills, state_size=24, hidden_size=64):
        super().__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size
        self.skills = nn.ModuleDict(pretrained_skills)

        for name, skill in self.skills.items():
            if not hasattr(skill, 'forward'):
                raise ValueError(f"Skill {name} must be a nn.Module with forward method")

        self.controller = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, len(pretrained_skills)),
            nn.Softmax(dim=-1)
        )

        self.arbiter = SimpleActionArbiter()

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        weights_tensor = self.controller(x)

        skill_names = list(self.skills.keys())

        input_state_all = torch.cat([
            x[:, :16],
            x[:, -8:]
        ], dim=1)  # [batch_size, 24]

        actions_dict = {}
        for skill_name, skill in self.skills.items():
            if skill_name != "enemy":
                action_mean = skill(input_state_all.detach())
            else:
                action_mean = skill(x.detach())

            actions_dict[skill_name] = action_mean

        weights_dict_batch = {}
        for idx, name in enumerate(skill_names):
            weights_dict_batch[name] = weights_tensor[:, idx]
        combined_action = self.arbiter.combine_batch(actions_dict, weights_dict_batch)

        return combined_action

    def get_params_vector(self) -> torch.Tensor:
        """Возвращает вектор параметров (уже на нужном устройстве)"""
        if self.params_vector is None:
            self.update_params_vector()
        return self.params_vector

    def update_params_vector(self):
        """Обновляет вектор параметров из весов сети"""
        params_list = []
        for param in self.parameters():
            params_list.append(param.data.view(-1))

        if params_list:
            self.params_vector = torch.cat(params_list).clone()
        else:
            self.params_vector = torch.tensor([], device=self.device)

    def set_params_from_vector(self, vector: torch.Tensor):
        """Устанавливает параметры сети из вектора"""
        idx = 0
        for param in self.parameters():
            size = param.data.numel()
            shape = param.data.shape
            # Извлекаем параметры из вектора
            param_data = vector[idx:idx + size].view(shape)
            param.data.copy_(param_data)
            idx += size
        self.params_vector = vector.clone()

