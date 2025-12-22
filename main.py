import Simulation
import Evolution
import BasicTrain
import Visualizer
import torch
import DDPG
import TD3


def start_evolution_algorithm(env):
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


def start_basic_algorithm(env, count=10):
    # Создаем и запускаем эволюционный алгоритм
    ba = BasicTrain.BasicAlgorithm(env=env, input_size=10, device="cuda")

    # Запускаем обучение
    for i in range(count):
        best_individual = ba.run(epochs=100, count_steps=500, name=i)


def start_combined_algorithm(env, population_size=10):
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


def start_ddpg(env, epoch=100):
    ddpg = DDPG.DDPG(gamma=0.99, tau=0.003, hidden_size=256, state_size=30, action_size=2, batch_size=32, device="cuda", count_last_states=1)
    ddpg.train(env=env, epochs=epoch, count_steps=300)


def start_td3(env, epoch=100):
    td3 = TD3.TD3(gamma=0.5, tau=0.001, hidden_size=256, state_size=30, action_size=2, batch_size=32, device="cuda", count_last_states=1, policy_update_freq=2, policy_noise=0.1)
    td3.load_actor("sim2_model/DDPG_10000_-10.1044.pth")
    td3.train(env=env, epochs=epoch, count_steps=300)


def save_model_run(env, path, count_steps):
    network = torch.load(path, weights_only=False)
    individual = Evolution.Individual(network, device="cuda")
    Visualizer.animation(individual=individual, env=env, device="cuda", render=True, count_steps=count_steps)


if __name__ == '__main__':
    env = Simulation.Environment2(
        field_size=50.0
    )

    # start_basic_algorithm(env)
    # start_evolution_algorithm(env)
    # start_combined_algorithm(env, population_size=10)
    # start_ddpg(env, 10001)
    start_td3(env, 10001)
    # save_model_run(env=env, path="pretraining_model/DDPG_1500612.3310.pth", count_steps=1000)
