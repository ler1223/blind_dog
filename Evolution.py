import copy

import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp
from typing import List, Tuple
import pickle
import json
from scipy.spatial.distance import cdist
import Simulation
import Visualizer


class EvolutionaryNetwork(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=2):
        super().__init__()

        # Кодируем архитектуру в векторе параметров
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()
        )

        # Храним параметры как вектор для эволюции
        self.params_vector = None
        self.update_params_vector()

    def forward(self, x):
        return self.net(x)

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


class Individual:
    """
    Индивидуум (собака) в эволюционном алгоритме
    """

    def __init__(self, network: EvolutionaryNetwork, dna=None,  device="cpu"):
        self.device = device
        self.network = network.to(self.device)
        self.network.params_vector = None
        self.dna = dna if dna is not None else network.get_params_vector()
        self.dna.to(device=self.device)
        self.fitness = 0.0
        self.age = 0
        self.species_id = -1
        self.stats = {
            'targets_reached': 0,
            'avg_speed': 0.0,
            'collisions': 0,
            'distance_traveled': 0.0
        }
        self.network.set_params_from_vector(self.dna)

    def evaluate(self, simulation_env, num_episodes=3, count_steps=300):
        """
        Оценка фитнеса индивидуума
        """
        total_fitness = 0.0

        for episode in range(num_episodes):
            fitness, stats = self._run_episode(simulation_env, count_steps)
            total_fitness += fitness

            # Обновляем статистику
            for key in self.stats:
                self.stats[key] += stats[key]

        self.fitness = total_fitness / num_episodes
        # Нормализуем статистику
        for key in self.stats:
            self.stats[key] /= num_episodes

        return self.fitness

    def predict(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        predict = self.network.forward(state)
        if self.device == "cuda":
            predict = predict.cpu()
        accelerate = predict.squeeze(0)
        accelerate, angle_acceleration = accelerate[0], accelerate[1]
        return accelerate, angle_acceleration

    def _run_episode(self, env, count_steps) -> Tuple[float, dict]:
        dog, target = env.reset()
        fitness = 0
        for i in range(count_steps):
            with torch.no_grad():
                state = env.get_state(dog, target)
                accelerate, angle_acceleration = self.predict(state)
            dog, targets, reward, done = env.step(dog, target, accelerate, angle_acceleration)
            fitness += reward

        stats = {
            'targets_reached': 0,
            'avg_speed': 0.0,
            'collisions': 0,
            'distance_traveled': 0.0
        }

        return fitness, stats

    def mutate(self, mutation_rate=0.01, mutation_strength=0.01):
        """
        Мутация DNA индивидуума с использованием PyTorch
        """
        with torch.no_grad():
            new_dna = self.dna.clone()

            mask = torch.rand_like(new_dna) < mutation_rate
            mutation = torch.randn_like(new_dna) * mutation_strength
            new_dna[mask] += mutation[mask]

            # Ограничиваем значения
            new_dna = torch.clamp(new_dna, -1, 1)

        return Individual(self.network, new_dna, device=self.device)

    def crossover(self, other: 'Individual') -> 'Individual':
        """
        Скрещивание двух индивидуумов с использованием PyTorch
        """
        with torch.no_grad():
            mask = torch.rand_like(self.dna) > 0.5
            child_dna = self.dna.clone()
            child_dna[mask] = other.dna[mask]

        return Individual(self.network, child_dna, device=self.device)


class Species:
    """
    Вид в эволюционном алгоритме (NEAT концепция)
    """

    def __init__(self, id: int, representative: Individual, device="cpu"):
        self.id = id
        self.representative = representative
        self.device = device
        self.members: List[Individual] = [representative]
        self.best_fitness = representative.fitness
        self.stagnation = 0
        self.adjusted_fitness_sum = 0.0

    def add_member(self, individual: Individual):
        individual.species_id = self.id
        self.members.append(individual)

        if individual.fitness > self.best_fitness:
            self.best_fitness = individual.fitness
            self.stagnation = 0
        else:
            self.stagnation += 1

    def calculate_adjusted_fitness(self):
        """Рассчитывает скорректированный фитнес"""
        self.adjusted_fitness_sum = sum(m.fitness for m in self.members) / len(self.members)

    def cull(self, survival_rate=0.5):
        """Удаляет худших особей"""
        self.members.sort(key=lambda x: x.fitness, reverse=True)
        keep_count = max(2, int(len(self.members) * survival_rate))
        self.members = self.members[:keep_count]

    def reproduce(self, offspring_count: int) -> List[Individual]:
        """Размножение внутри вида"""
        offspring = []

        # Элитизм: сохраняем лучших
        self.members.sort(key=lambda x: x.fitness, reverse=True)
        elite_count = min(2, len(self.members))
        offspring.extend(self.members[:elite_count])

        # Скрещивание и мутация
        for _ in range(offspring_count - elite_count):
            if len(self.members) >= 2:
                # Выбор родителей (турнирная селекция)
                parents = np.random.choice(self.members, size=2, replace=False)
                parent1, parent2 = sorted(parents, key=lambda x: x.fitness, reverse=True)[:2]

                # Скрещивание
                if np.random.random() < 0.5:
                    child = parent1.crossover(parent2)
                else:
                    child = Individual(parent1.network, parent1.dna.clone(), device=self.device)

                # Мутация
                if np.random.random() < 0.4:
                    mutation_rate = np.random.uniform(0.05, 0.1)
                    mutation_strength = np.random.uniform(0.05, 0.1)
                    child = child.mutate(mutation_rate, mutation_strength)
            else:
                # Клонирование с мутацией
                parent = self.members[0]
                child = parent.mutate(mutation_rate=0.3, mutation_strength=0.3)

            child.age = 0
            offspring.append(child)

        return offspring


class EvolutionaryAlgorithm:
    """
    Основной класс эволюционного алгоритма (гибрид NEAT + CMA-ES)
    """

    def __init__(self,
                 network_template,
                 population_size=100,
                 generations=100,
                 steps_simulation=300,
                 interval_animation=100,
                 compatibility_threshold=3.0,
                 species_target=10,
                 device='cpu',
                 render=True,
                 init_population=None,
                 path_save_anim=None,
                 path_save_model=None):

        self.population_size = population_size
        self.generations = generations
        self.steps_simulation = steps_simulation
        self.interval_animation = interval_animation
        self.compatibility_threshold = compatibility_threshold
        self.species_target = species_target
        self.device = device
        self.render = render
        self.path_save_anim = path_save_anim
        self.path_save_model = path_save_model

        # Инициализация популяции
        self.network_template = network_template
        self.population: List[Individual] = []
        self.species: List[Species] = []
        self.generation = 0

        # Статистика
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.species_count_history = []
        self.init_population = init_population

        # Инициализируем популяцию
        self._initialize_population(self.init_population)

    def _initialize_population(self, init_population):
        """Инициализация начальной популяции"""
        print("Initializing population...")

        if init_population is not None:
            for individual in init_population:
                self.population.append(individual)
        for i in range(self.population_size - len(self.population)):
            # Создаем нового индивидуума
            network = EvolutionaryNetwork(
                self.network_template.net[0].in_features,
                self.network_template.net[0].out_features,
                self.network_template.net[-2].out_features
            )

            # Случайная инициализация весов
            dna = torch.randn_like(network.get_params_vector()) * 0.1
            individual = Individual(network, dna, self.device)

            self.population.append(individual)

        # Создаем начальные виды
        self._speciate()

    def _genetic_distance(self, ind1: Individual, ind2: Individual) -> float:
        """
        Вычисляет генетическое расстояние между двумя индивидуумами
        (упрощенная версия NEAT расстояния)
        """
        # Евклидово расстояние между DNA векторами
        dna_distance = torch.norm(ind1.dna.to(self.device) - ind2.dna.to(self.device))

        # Нормализуем по размеру DNA
        normalized_distance = dna_distance / np.sqrt(len(ind1.dna))

        return normalized_distance

    def _speciate(self):
        """Разделение популяции на виды"""
        self.species.clear()

        if not self.population:
            return

        # Создаем первый вид с первым индивидуумом
        first_species = Species(0, self.population[0], device=self.device)
        self.species.append(first_species)

        # Распределяем остальных индивидуумов по видам
        for individual in self.population[1:]:
            placed = False

            # Пробуем поместить в существующий вид
            for species in self.species:
                distance = self._genetic_distance(individual, species.representative)
                if distance < self.compatibility_threshold:
                    species.add_member(individual)
                    placed = True
                    break

            # Если не подошел ни к одному виду, создаем новый
            if not placed:
                new_species = Species(len(self.species), individual, device=self.device)
                self.species.append(new_species)

        # Удаляем пустые виды
        self.species = [s for s in self.species if s.members]

        # Обновляем порог совместимости для поддержания целевого количества видов
        if len(self.species) < self.species_target:
            self.compatibility_threshold *= 0.95
        elif len(self.species) > self.species_target:
            self.compatibility_threshold *= 1.05

        self.compatibility_threshold = np.clip(self.compatibility_threshold, 1.0, 10.0)

    def _evaluate_population(self, simulation_env, parallel=True):
        """Оценка фитнеса популяции"""
        if parallel:
            self._evaluate_parallel(simulation_env)
        else:
            self._evaluate_sequential(simulation_env)

        # Обновляем лучших представителей видов
        for species in self.species:
            if species.members:
                species.representative = max(species.members, key=lambda x: x.fitness)
                species.best_fitness = species.representative.fitness

    def _evaluate_sequential(self, simulation_env):
        """Последовательная оценка"""
        for individual in self.population:
            individual.evaluate(simulation_env, num_episodes=3, count_steps=self.steps_simulation)

    def _evaluate_parallel(self, simulation_env):
        """Параллельная оценка с использованием multiprocessing"""
        print("Parallel evaluation not implemented, using sequential...")
        self._evaluate_sequential(simulation_env)

    def _natural_selection(self):
        """Естественный отбор и создание нового поколения"""
        # Удаляем старые виды
        max_stagnation = 100
        self.species = [s for s in self.species if s.stagnation < max_stagnation]

        # Рассчитываем скорректированный фитнес
        for species in self.species:
            species.calculate_adjusted_fitness()

        # Вычисляем общий скорректированный фитнес
        total_adjusted_fitness = sum(s.adjusted_fitness_sum for s in self.species)

        # Распределение потомков между видами
        new_population = []

        for species in self.species:
            if total_adjusted_fitness > 0:
                # Количество потомков пропорционально вкладу вида
                offspring_count = int(
                    (species.adjusted_fitness_sum / total_adjusted_fitness) *
                    self.population_size
                )
            else:
                offspring_count = 1

            offspring_count = max(2, offspring_count)  # Минимум 2 потомка на вид

            # Отбор внутри вида (удаляем худших)
            species.cull(survival_rate=0.3)

            # Размножение
            offspring = species.reproduce(offspring_count)
            new_population.extend(offspring)

        # Если популяция меньше нужного размера, добавляем случайных индивидуумов
        while len(new_population) < self.population_size:
            parent = np.random.choice(self.population)
            child = parent.mutate(mutation_rate=0.2, mutation_strength=0.2)
            new_population.append(child)

        # Обрезаем если больше
        new_population = new_population[:self.population_size]

        # Обновляем популяцию
        self.population = new_population

        # Увеличиваем возраст всех индивидуумов
        for individual in self.population:
            individual.age += 1

    def _collect_statistics(self):
        """Сбор статистики по поколению"""
        if not self.population:
            return

        best_individual = max(self.population, key=lambda x: x.fitness)
        avg_fitness = np.mean([ind.fitness for ind in self.population])

        self.best_fitness_history.append(best_individual.fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.species_count_history.append(len(self.species))

        print(f"\n=== Generation {self.generation} ===")
        print(f"Best fitness: {best_individual.fitness:.2f}")
        print(f"Average fitness: {avg_fitness:.2f}")
        print(f"Species count: {len(self.species)}")
        print(f"Targets reached (best): {best_individual.stats['targets_reached']:.1f}")
        print(f"Collisions (best): {best_individual.stats['collisions']:.1f}")

    def run(self, simulation_env, checkpoint_interval=10):
        """
        Запуск эволюционного алгоритма
        """
        print(f"\nStarting Evolutionary Algorithm")
        print(f"Population: {self.population_size}, Generations: {self.generations}")

        for gen in range(self.generations):
            self.generation = gen

            # 1. Оценка фитнеса
            print(f"\nGeneration {gen}: Evaluating population...")
            self._evaluate_population(simulation_env, parallel=False)
            for i in self.population:
                print(i.fitness)

            # 2. Сбор статистики
            self._collect_statistics()

            # 3. Сохранение checkpoint
            if gen % checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_gen_{gen}.pkl')

            # 4. Естественный отбор (кроме последнего поколения)
            if gen < self.generations - 1:
                print("Natural selection...")
                self._natural_selection()

                # 5. Видообразование
                self._speciate()
            best_individual = max(self.population, key=lambda x: x.fitness)
            if gen % 100 == 0:
                if self.path_save_anim is None:
                    path = "anim_generation"
                else:
                    path = self.path_save_anim
                Visualizer.animation(best_individual, env=simulation_env, device="cuda",
                                     render=False, save_path=path + "/DDPG_epoch_" + str(gen))

                if self.path_save_model is None:
                    path = "pretraining_model"
                else:
                    path = self.path_save_model
                torch.save(best_individual.network, f"./{path}/DDPG_" + str(gen) + ".pth")

        # Возвращаем лучшего индивидуума
        best_individual = max(self.population, key=lambda x: x.fitness)
        return best_individual

    def save_checkpoint(self, filename):
        """Сохранение checkpoint"""
        checkpoint = {
            'generation': self.generation,
            'population': self.population,
            'species': self.species,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'species_count_history': self.species_count_history,
            'compatibility_threshold': self.compatibility_threshold
        }

        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        """Загрузка checkpoint"""
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)

        self.generation = checkpoint['generation']
        self.population = checkpoint['population']
        self.species = checkpoint['species']
        self.best_fitness_history = checkpoint['best_fitness_history']
        self.avg_fitness_history = checkpoint['avg_fitness_history']
        self.species_count_history = checkpoint['species_count_history']
        self.compatibility_threshold = checkpoint['compatibility_threshold']

        print(f"Checkpoint loaded: {filename}, Generation: {self.generation}")