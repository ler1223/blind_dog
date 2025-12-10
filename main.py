import Simulation
import Evolution
import BasicTrain
import Visualizer
import torch


def start_evolution_algorithm():
    env = Simulation.Environment(
        field_size=50.0,
        num_targets=50,
        max_episode_steps=200,
        seed=42
    )

    # Создаем и запускаем эволюционный алгоритм
    ea = Evolution.EvolutionaryAlgorithm(
        population_size=100,
        generations=100,
        input_size=8,
        output_size=2,
        interval_animation=10,
        render=True,
        device="cuda"
    )

    # Запускаем обучение
    best_individual = ea.run(env, checkpoint_interval=10)

    print(f"\nBest individual fitness: {best_individual.fitness:.2f}")
    print(f"Targets reached: {best_individual.stats['targets_reached']:.1f}")
    print(f"Collisions: {best_individual.stats['collisions']:.1f}")


def start_basic_algorithm():
    env = Simulation.Environment(
        field_size=50.0,
        num_targets=50,
        max_episode_steps=200,
        seed=42
    )

    # Создаем и запускаем эволюционный алгоритм
    ba = BasicTrain.BasicAlgorithm(env=env, input_size=10, device="cuda")

    # Запускаем обучение
    best_individual = ba.run(epochs=100, count_steps=500)


if __name__ == '__main__':
    start_basic_algorithm()

