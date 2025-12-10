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
import Evolution
import math


class BasicAlgorithm:

    def __init__(self, env, input_size=8, output_size=2, loss_fn=torch.nn.CrossEntropyLoss(), device="cpu", interval_animation=30):
        self.input_size = input_size
        self.output_size = output_size
        self.env = env
        self.loss_fn = loss_fn
        self.device = device
        self.interval_animation = interval_animation

    @staticmethod
    def get_current_accelerate(dog: Simulation.Dog, target):
        vect = target.position - dog.position
        vect = vect / torch.norm(vect, dim=0)
        distant = torch.norm(vect)
        current_accelerate = torch.tanh(distant)

        current_accelerate_angle = torch.atan2(vect[1], vect[0]) - dog.angle
        alignment_factor = torch.cos(current_accelerate_angle).clamp(0, 1)
        current_accelerate_angle = (current_accelerate_angle + math.pi) % (2 * math.pi) - math.pi
        return torch.tensor((current_accelerate * alignment_factor, torch.tanh(current_accelerate_angle)))

    def animation(self, individual, count_steps):
        """Демонстрация анимации."""
        print("\n=== Simple Animation Demo ===")
        dog, target = self.env.reset()

        # Создаем анимацию
        animation = Visualizer.SimpleAnimation(
            env=self.env,
            dog=dog,
            individual=individual,
            target=target,
            steps=count_steps,
            interval=self.interval_animation,
            device=self.device
        )

        print("Animation created. Showing window...")
        animation.show()

    def run(self, epochs=10, count_steps=200):
        network = Evolution.EvolutionaryNetwork(self.input_size, 128, self.output_size)
        individual = Evolution.Individual(network, device=self.device)
        optimizer = torch.optim.Adam(individual.network.parameters(), lr=0.0001)
        loss_fn = torch.nn.MSELoss().to(self.device)

        dataset_states = []
        dataset_targets = []

        for epoch in range(epochs):
            dog, target = self.env.reset()
            for i in range(count_steps):
                state = self.env.get_state(dog, target)
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                current_accelerate = self.get_current_accelerate(dog, target)
                dog, target, reward = self.env.step(dog, target, current_accelerate[0], current_accelerate[1])
                dataset_states.append(state)
                dataset_targets.append(current_accelerate.unsqueeze(0).to(device=self.device))

        states_tensor = torch.cat(dataset_states, dim=0)
        targets_tensor = torch.cat(dataset_targets, dim=0)
        batch_size = 32
        num_batches = len(dataset_states) // batch_size
        print(num_batches)

        for epoch in range(epochs):
            epoch_loss = 0
            indices = torch.randperm(len(dataset_states))
            states_shuffled = states_tensor[indices]
            targets_shuffled = targets_tensor[indices]
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_states = states_shuffled[start_idx:end_idx].to(self.device)
                batch_targets = targets_shuffled[start_idx:end_idx].to(self.device)
                optimizer.zero_grad()

                predictions = individual.network.forward(batch_states)

                loss = loss_fn(predictions, batch_targets)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            print(avg_loss)
            self.animation(individual, count_steps*4)

        return individual
