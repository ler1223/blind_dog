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
        network_template=Evolution.EvolutionaryNetwork(10, 128, 2),
        population_size=10,
        generations=100,
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
    for i in range(10):
        best_individual = ba.run(epochs=100, count_steps=500, name=i)


def start_combined_algorithm(population_size=10):
    env = Simulation.Environment(
        field_size=50.0,
        num_targets=50,
        max_episode_steps=300,
        seed=42
    )

    # Создаем и запускаем эволюционный алгоритм
    # ba = BasicTrain.BasicAlgorithm(env=env, input_size=10, device="cuda")

    init_individuals = []
    for i in range(population_size):
        # best_individual = ba.run(epochs=100, count_steps=500, name=i)
        network = Evolution.EvolutionaryNetwork(10, 128, 2)
        network.load_state_dict(torch.load("pretraining_model/"+str(i)+"_state_dict.pth", weights_only=True))
        best_individual = Evolution.Individual(network, device="cuda")
        init_individuals.append(best_individual)

    ea = Evolution.EvolutionaryAlgorithm(
        init_individuals[0].network,
        population_size=200,
        generations=100,
        steps_simulation=300,
        interval_animation=10,
        render=False,
        device="cuda",
        init_population=init_individuals,
        save_animation_gen_dir="anim_generation"
    )

    # Запускаем обучение
    best_individual = ea.run(env, checkpoint_interval=1000)

    print(f"\nBest individual fitness: {best_individual.fitness:.2f}")
    print(f"Targets reached: {best_individual.stats['targets_reached']:.1f}")
    print(f"Collisions: {best_individual.stats['collisions']:.1f}")


if __name__ == '__main__':
    # start_basic_algorithm()
    # start_evolution_algorithm()
    start_combined_algorithm(population_size=10)
