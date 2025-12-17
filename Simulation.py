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
    Моделирует поле с собакой целью и физикой взаимодействия.
    """

    def __init__(self,
                 field_size=10.0,
                 seed=None):

        self.field_size = field_size  # Размер квадратного поля (от -field_size/2 до field_size/2)
        self.seed = seed
        self.velocity_damping = 0.98
        self.angular_damping = 0.95
        self.last_position = None

        # Инициализация генератора случайных чисел
        if seed is not None:
            np.random.seed(seed)

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
            dog: Объект собаки с начальным состоянием.
            target: цель.
        """
        # Создаем собаку
        dog = Dog(position=(dog_position if dog_position is not None else self._random_position()), size=dog_size)
        self.last_position = None
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
            target: Обновленная цель.
            reward: Награда за шаг.
            done: всегда True
        """
        dog.velocity *= self.velocity_damping
        dog.angle_velocity *= self.angular_damping

        dog.angle_acceleration = angle_acceleration
        dog.angle_velocity += dog.angle_acceleration * 0.5
        dog.angle_velocity = torch.clip(dog.angle_velocity, -0.7, 0.7)
        dog.angle += dog.angle_velocity * 0.5
        dog.angle = (dog.angle + math.pi) % (2 * math.pi) - math.pi

        dog.acceleration = torch.cat((torch.cos(dog.angle) * acceleration, torch.sin(dog.angle) * acceleration))

        distance_to_target = np.linalg.norm(target.position - dog.position)

        # Применяем ускорение (с ограничением)
        dog.acceleration = torch.clip(dog.acceleration, -1, 1) * 2.0

        # Обновляем скорость
        dog.velocity += dog.acceleration * 0.5

        # Ограничиваем максимальную скорость
        speed = np.linalg.norm(dog.velocity)
        if speed > dog.max_speed:
            dog.velocity = (dog.velocity / speed) * dog.max_speed

        # Обновляем позицию
        dog.position += dog.velocity * 0.5

        # Проверяем столкновения с границами
        collision = self._check_boundary_collision(dog)
        target_reached = False

        reward = -0.2

        if target.active:
            distance = np.linalg.norm(target.position - dog.position)
            if distance < (dog.size + target.radius):
                target_reached = True
            else:
                # Награда за приближение
                if self.last_position is not None:
                    old_distance = np.linalg.norm(target.position - self.last_position)
                    distance_reward = (old_distance - distance_to_target) * 0.1
                    reward += max(distance_reward, -0.1) - distance / self.field_size

        self.last_position = dog.position.clone()

        if target_reached:
            reward += 50.0
            target = self.create_target(target.id + 1)

        # Штраф за столкновение с границей
        if collision:
            reward -= 5.0 * np.linalg.norm(dog.velocity)
        done = False
        return dog, target, reward, done

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
        """
        # Нормализация
        half_field = self.field_size / 2

        # Позиция и скорость
        dog_pos = dog.position.numpy() / half_field
        dog_vel = dog.velocity.numpy() / dog.max_speed

        # Угол (нормализация от -1 до 1)
        angle_sin = math.sin(dog.angle.item())
        angle_cos = math.cos(dog.angle.item())

        # Угловая скорость (нормализация)
        norm_angle_vel = dog.angle_velocity.item() / 0.7  # Макс. угловая скорость 0.7

        # Позиция цели
        target_pos = target.position.numpy() / half_field

        # Вектор к цели (относительный)
        to_target = target.position - dog.position
        norm_to_target = to_target.numpy() / self.field_size  # Нормализация

        # Расстояние до цели (нормализованное)
        distance = torch.norm(to_target).item() / self.field_size

        # Расстояние до границ
        boundary_x = (half_field - abs(dog.position[0].item())) / half_field
        boundary_y = (half_field - abs(dog.position[1].item())) / half_field

        return np.array([
            dog_pos[0], dog_pos[1],
            dog_vel[0], dog_vel[1],
            angle_sin, angle_cos,
            norm_angle_vel,
            target_pos[0], target_pos[1],
            norm_to_target[0], norm_to_target[1],
            distance,
            boundary_x, boundary_y
        ], dtype=np.float32)


class Environment2:

    class Enemy:
        def __init__(self, start_position=torch.zeros(2), size=2, velocity=torch.ones(2)):
            self.position = start_position
            self.size = size
            self.velocity = velocity

    def __init__(self,
                 field_size=10.0,
                 count_enemy=4,
                 seed=None):

        self.field_size = field_size  # Размер квадратного поля (от -field_size/2 до field_size/2)
        self.seed = seed
        self.velocity_damping = 0.98
        self.angular_damping = 0.95
        self.last_position = None
        self.enemies = []
        self.count_enemy = count_enemy
        self.put_last_target = 0
        self.states = []

        # Инициализация генератора случайных чисел
        if seed is not None:
            np.random.seed(seed)

    def create_target(self, i):
        target = type('Target', (), {})()
        target.position = self._random_position()
        target.radius = 0.3
        target.active = True
        target.id = i
        return target

    def create_enemy(self, dog):
        position = self._random_position()
        while np.linalg.norm(dog.position - position) < self.field_size / 5:
            position = self._random_position()
        velocity = torch.rand(2)
        velocity = velocity / torch.norm(velocity)
        return self.Enemy(start_position=position, velocity=velocity)

    def reset(self, dog_position=None, dog_size=0.5):
        """
        Сброс среды для нового эпизода.

        Args:
            dog_position: Начальная позиция собаки. Если None - случайная позиция. От -4/5 до 4/5 от размера поля.
            dog_size: Размер собаки (радиус).

        Returns:
            dog: Объект собаки с начальным состоянием.
            target: цель.
        """
        # Создаем собаку
        self.states = []
        dog = Dog(position=(dog_position if dog_position is not None else self._random_position()), size=dog_size)
        self.last_position = None
        # Создаем цели
        target = self.create_target(0)
        self.put_last_target = 0
        self.enemies = []
        for i in range(self.count_enemy):
            self.enemies.append(self.create_enemy(dog))
        return dog, target

    def step(self, dog, target, acceleration, angle_acceleration, step, max_step):
        """
        Выполняет один шаг симуляции.

        Args:
            dog: Объект собаки.
            target: цель.
            acceleration: Ускорение (torch.tensor[1]).
            angle_acceleration: Ускорение вращения (torch.tensor[1]).

        Returns:
            dog: Обновленный объект собаки.
            target: Обновленная цель.
            reward: Награда за шаг.
            done: всегда True
        """
        dog.velocity *= self.velocity_damping
        dog.angle_velocity *= self.angular_damping

        dog.angle_acceleration = angle_acceleration
        dog.angle_velocity += dog.angle_acceleration * 0.5
        dog.angle_velocity = torch.clip(dog.angle_velocity, -0.7, 0.7)
        dog.angle += dog.angle_velocity * 0.5
        dog.angle = (dog.angle + math.pi) % (2 * math.pi) - math.pi

        dog.acceleration = torch.cat((torch.cos(dog.angle) * acceleration, torch.sin(dog.angle) * acceleration))

        distance_to_target = np.linalg.norm(target.position - dog.position)

        # Применяем ускорение (с ограничением)
        dog.acceleration = torch.clip(dog.acceleration, -1, 1) * 2.0

        # Обновляем скорость
        dog.velocity += dog.acceleration * 0.5

        # Ограничиваем максимальную скорость
        speed = np.linalg.norm(dog.velocity)
        if speed > dog.max_speed:
            dog.velocity = (dog.velocity / speed) * dog.max_speed

        # Обновляем позицию
        dog.position += dog.velocity * 0.5

        # Проверяем столкновения с границами
        collision = self._check_boundary_collision(dog)
        target_reached = False

        collision_enemies = self.update_enemies(dog)

        reward = -0.01 + 0.00005*step

        if target.active:
            distance = np.linalg.norm(target.position - dog.position)
            if distance < (dog.size + target.radius):
                target_reached = True
            else:
                # Награда за приближение
                if self.last_position is not None:
                    old_distance = np.linalg.norm(target.position - self.last_position)
                    distance_reward = (old_distance - distance_to_target) * 0.1
                    reward += max(distance_reward, -0.1) - distance / self.field_size

        self.last_position = dog.position.clone()

        if target_reached:
            reward += 20.0
            # reward += 40
            target = self.create_target(target.id + 1)
            self.put_last_target = step

        # Штраф за столкновение с границей
        if collision:
            reward -= 5.0 * np.linalg.norm(dog.velocity)

        done = False
        if collision_enemies:
            reward += -60
            done = True
        return dog, target, reward, done

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

    def update_enemies(self, dog):
        collision_dog = False
        for enemy in self.enemies:
            enemy.position += enemy.velocity * 0.5

            half_size = self.field_size / 2
            # Проверка по X
            if enemy.position[0] - enemy.size < -half_size:
                enemy.position[0] = -half_size + enemy.size
                enemy.velocity[0] *= -1
            elif enemy.position[0] + enemy.size > half_size:
                enemy.position[0] = half_size - enemy.size
                enemy.velocity[0] *= -1

            # Проверка по Y
            if enemy.position[1] - enemy.size < -half_size:
                enemy.position[1] = -half_size + enemy.size
                enemy.velocity[1] *= -1
            elif enemy.position[1] + enemy.size > half_size:
                enemy.position[1] = half_size - enemy.size
                enemy.velocity[1] *= -1

            if torch.abs(enemy.position[0] - dog.position[0]) < (enemy.size + dog.size) and \
                    torch.abs(enemy.position[1] - dog.position[1]) < (enemy.size + dog.size):
                collision_dog = True

        return collision_dog

    def get_enemy_position(self, dog, count=4):
        enemies_position = []
        for enemy in self.enemies:
            enemy_pos = enemy.position.numpy() / self.field_size / 2
            # Вектор к цели (относительный)
            to_enemy = enemy.position - dog.position
            norm_to_enemy = to_enemy.numpy() / self.field_size  # Нормализация
            distance = torch.norm(to_enemy).item() / self.field_size
            enemies_position.append((enemy_pos, norm_to_enemy, distance, enemy.velocity / dog.max_speed))

        result_enemies = sorted(enemies_position, key=lambda x: x[2])[:count]
        result = []
        for pos, dir_vec, dist, vel in result_enemies:
            result.extend([pos[0], pos[1], dir_vec[0], dir_vec[1], vel[0], vel[1], dist])
        return np.array(result, dtype=np.float32)

    def get_state(self, dog, target, count_last_states=1, save=True):
        """
        Получает вектор состояния для нейронной сети.
        """
        # Нормализация
        half_field = self.field_size / 2

        # Позиция и скорость
        dog_pos = dog.position.numpy() / half_field
        dog_vel = dog.velocity.numpy() / dog.max_speed

        # Угол (нормализация от -1 до 1)
        angle_sin = math.sin(dog.angle.item())
        angle_cos = math.cos(dog.angle.item())

        # Угловая скорость (нормализация)
        norm_angle_vel = dog.angle_velocity.item() / 0.7  # Макс. угловая скорость 0.7

        # Позиция цели
        target_pos = target.position.numpy() / half_field

        # Вектор к цели (относительный)
        to_target = target.position - dog.position
        norm_to_target = to_target.numpy() / self.field_size  # Нормализация

        enemies_position = torch.FloatTensor(self.get_enemy_position(dog, self.count_enemy))

        # Расстояние до цели (нормализованное)
        distance = torch.norm(to_target).item() / self.field_size

        # Расстояние до границ
        boundary_x = (half_field - abs(dog.position[0].item())) / half_field
        boundary_y = (half_field - abs(dog.position[1].item())) / half_field

        result = torch.FloatTensor([
            dog_pos[0], dog_pos[1],
            dog_vel[0], dog_vel[1],
            angle_sin, angle_cos,
            norm_angle_vel,
            target_pos[0], target_pos[1],
            norm_to_target[0], norm_to_target[1],
            distance,
            boundary_x, boundary_y
        ])

        state = torch.cat([result, enemies_position]).unsqueeze(0)

        if count_last_states > 1 and len(self.states) > 0:
            last_states = torch.cat(self.states[-count_last_states + 1:], dim=0)
            result = torch.cat([state, last_states], dim=0)
        else:
            result = state
        if save:
            self.states.append(state)
        return result
