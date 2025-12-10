import math

import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp
from typing import List, Tuple
import pickle
import json
from scipy.spatial.distance import cdist


class Dog:
    def __init__(self, position=torch.zeros(2), velocity=torch.zeros(2), acceleration=torch.zeros(2),
                 angle=torch.zeros(1), angle_velocity=torch.zeros(1), angle_acceleration=torch.zeros(1),
                 size=0.5, max_speed=5):

        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.angle = angle
        self.angle_velocity = angle_velocity
        self.angle_acceleration = angle_acceleration
        self.size = size
        self.max_speed = max_speed


class Environment:
    """
    Среда для эволюционного алгоритма.
    Моделирует поле с собаками, целями и физикой взаимодействий.
    """

    def __init__(self,
                 field_size=10.0,
                 num_targets=5,
                 max_episode_steps=500,
                 seed=None):

        self.field_size = field_size  # Размер квадратного поля (от -field_size/2 до field_size/2)
        self.num_targets = num_targets
        self.max_episode_steps = max_episode_steps
        self.seed = seed

        # Инициализация генератора случайных чисел
        if seed is not None:
            np.random.seed(seed)

        # Статистика среды
        self.stats = {
            'total_episodes': 0,
            'total_targets_reached': 0,
            'total_collisions': 0
        }

    def create_target(self, i):
        target = type('Target', (), {})()
        target.position = self._random_position()
        target.radius = 0.3
        target.active = True
        target.id = i
        return target

    def reset(self, dog_position=None, dog_size=0.5):
        """
        Сброс среды для нового эпизода.

        Args:
            dog_position: Начальная позиция собаки. Если None - случайная позиция. От -4/5 до 4/5 от размера поля.
            dog_size: Размер собаки (радиус).

        Returns:
            dog_obj: Объект собаки с начальным состоянием.
            targets: Список целей.
        """
        # Создаем собаку
        dog = Dog(position=(dog_position if dog_position is not None else self._random_position()), size=dog_size)

        # Создаем цели
        target = self.create_target(0)

        return dog, target

    def step(self, dog, target, acceleration, angle_acceleration):
        """
        Выполняет один шаг симуляции.

        Args:
            dog: Объект собаки.
            target: цель.
            acceleration: Ускорение (torch.tensor[1]).
            angle_acceleration: Ускорение вращения (torch.tensor[1]).

        Returns:
            dog: Обновленный объект собаки.
            targets: Обновленная цель.
            reward: Награда за шаг.
        """

        dog.angle_acceleration = angle_acceleration
        dog.angle_velocity += dog.angle_acceleration
        dog.angle_velocity = torch.clip(dog.angle_velocity, -0.7, 0.7)
        dog.angle += dog.angle_velocity * 0.1
        dog.angle = (dog.angle + math.pi) % (2 * math.pi) - math.pi

        dog.acceleration = torch.cat((torch.cos(dog.angle) * acceleration, torch.sin(dog.angle) * acceleration))

        last_position = dog.position
        distance_to_target = np.linalg.norm(target.position - dog.position)

        # Применяем ускорение (с ограничением)
        dog.acceleration = torch.clip(dog.acceleration, -1, 1) * 2.0

        # Обновляем скорость
        dog.velocity += dog.acceleration

        # Ограничиваем максимальную скорость
        speed = np.linalg.norm(dog.velocity)
        if speed > dog.max_speed:
            dog.velocity = (dog.velocity / speed) * dog.max_speed

        # Обновляем позицию
        dog.position += dog.velocity * 0.1

        # Проверяем столкновения с границами
        collision = self._check_boundary_collision(dog)
        target_reached = False

        reward = 0.0

        if target.active:
            distance = np.linalg.norm(target.position - dog.position)
            if distance < (dog.size + target.radius):
                target_reached = True
            else:
                # Награда за приближение
                old_distance = np.linalg.norm(target.position - last_position)
                distance_reward = (old_distance - distance_to_target) * 0.1
                reward += max(distance_reward, -1.0)  # Минимальный штраф

        if target_reached:
            reward += 10.0
            target = self.create_target(target.id + 1)

        # Штраф за столкновение с границей
        if collision:
            reward -= 5.0 * np.linalg.norm(dog.velocity)

        # Обновляем статистику
        if target_reached:
            self.stats['total_targets_reached'] += 1
        if collision:
            self.stats['total_collisions'] += 1
        return dog, target, reward

    def _random_position(self):
        """Генерирует случайную позицию внутри поля."""
        half_size = self.field_size / 2
        return torch.FloatTensor(np.random.uniform(-half_size + half_size / 5, half_size - half_size / 5, 2))

    def _check_boundary_collision(self, dog):
        """
        Проверяет столкновение собаки с границами поля.
        Возвращает True если было столкновение.
        """
        half_size = self.field_size / 2

        collision = False

        # Проверка по X
        if dog.position[0] - dog.size < -half_size:
            dog.position[0] = -half_size + dog.size
            dog.velocity[0] *= -0.5  # Отскок с потерей энергии
            collision = True
        elif dog.position[0] + dog.size > half_size:
            dog.position[0] = half_size - dog.size
            dog.velocity[0] *= -0.5
            collision = True

        # Проверка по Y
        if dog.position[1] - dog.size < -half_size:
            dog.position[1] = -half_size + dog.size
            dog.velocity[1] *= -0.5
            collision = True
        elif dog.position[1] + dog.size > half_size:
            dog.position[1] = half_size - dog.size
            dog.velocity[1] *= -0.5
            collision = True

        return collision

    def get_state(self, dog, target):
        """
        Получает вектор состояния для нейронной сети.
        (Аналогично методу _get_state в классе Individual)
        """
        # Нормализованные координаты
        dog_pos = dog.position / (self.field_size / 2)
        dog_vel = dog.velocity / dog.max_speed

        # Расстояние до границ
        half_field = self.field_size / 2
        boundary_dist = np.array([
            (half_field - abs(dog.position[0])) / half_field,
            (half_field - abs(dog.position[1])) / half_field
        ])

        return np.concatenate([
            dog_pos,
            dog_vel,
            dog.angle,
            dog.angle_velocity,
            target.position / (self.field_size / 2),
            boundary_dist
        ])

    def render(self, dog, step=None):
        if step is not None:
            print(f"\nStep {step}:")
        print(f"  Dog position: ({dog.position[0]:.2f}, {dog.position[1]:.2f})")
        print(f"  Dog velocity: ({dog.velocity[0]:.2f}, {dog.velocity[1]:.2f})")
        print(f"  angle velocity: ({dog.angle_velocity[0]:.2f}, {dog.angle_velocity[1]:.2f})")

    def get_stats(self):
        """Возвращает статистику среды."""
        return self.stats.copy()