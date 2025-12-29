import Simulation
import Evolution
import BasicTrain
import Visualizer
import torch
import DDPG
import TD3
import PPO
import argparse


def start_evolution_algorithm(env):
    # Создаем и запускаем эволюционный алгоритм
    ea = Evolution.EvolutionaryAlgorithm(
        network_template=Evolution.EvolutionaryNetwork(10, 128, 2),
        population_size=10,
        generations=100,
        interval_animation=10,
        render=True,
        device="cuda",
        path_save_anim="evolve/anim",
        path_save_model="evolve/model"
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
        path_save_anim="evolve/anim",
        path_save_model="evolve/model"
    )

    # Запускаем обучение
    best_individual = ea.run(env, checkpoint_interval=1000)

    print(f"\nBest individual fitness: {best_individual.fitness:.2f}")
    print(f"Targets reached: {best_individual.stats['targets_reached']:.1f}")
    print(f"Collisions: {best_individual.stats['collisions']:.1f}")


def start_ddpg(env, epoch=100):
    print("start DDPG")
    state_size = env.get_state_size()
    ddpg = DDPG.DDPG(gamma=0.99, tau=0.003, hidden_size=256, state_size=state_size, action_size=2, batch_size=32, device="cuda", count_last_states=1, path_save_anim="DDPG/anim", path_save_model="DDPG/model")
    ddpg.train(env=env, epochs=epoch, count_steps=300)


def start_td3(env, epoch=100):
    print("start td3")
    state_size = env.get_state_size()
    td3 = TD3.TD3(gamma=0.5, tau=0.001, hidden_size=256, state_size=state_size, action_size=2, batch_size=32, device="cuda", count_last_states=1, policy_update_freq=2, policy_noise=0.1, path_save_anim="td3/anim", path_save_model="td3/model")
    # td3.load_actor("sim2_model/DDPG_10000_-10.1044.pth")
    td3.train(env=env, epochs=epoch, count_steps=300)


def save_model_run(env, path, count_steps):
    network = torch.load(path, weights_only=False)
    individual = Evolution.Individual(network, device="cuda")
    Visualizer.animation(individual=individual, env=env, device="cuda", render=True, count_steps=count_steps)


def start_td3_sim3(env, epoch=100):
    td3 = TD3.TD3(gamma=0.5, tau=0.001, hidden_size=512, state_size=32, action_size=2, batch_size=256, device="cuda", count_last_states=3, policy_update_freq=1, policy_noise=0.3, path_save_anim="td3/anim", path_save_model="td3/model")
    # td3.load_actor("sim3_2/DDPG_12900_0.4655.pth")
    td3.train(env=env, epochs=epoch, count_steps=400)


def start_PPO(env, epoch=100):
    print("start PPO")
    state_size = env.get_state_size()
    ppo = PPO.PPO(gamma=0.99, gae_lambda=0.95, hidden_size=512, state_size=state_size, action_size=2, batch_size=64, device="cuda", count_last_states=1, policy_update_freq=2, count_retrain=10, path_save_anim="PPO/anim", path_save_model="PPO/model")
    # td3.load_actor("sim3_2/DDPG_12900_0.4655.pth")
    ppo.train(env=env, epochs=epoch, count_steps=400)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Описание вашей программы')
    # Позиционные аргументы
    parser.add_argument('env', default='3', help='Версия симуляции [1, 2, 3]')

    # Опциональные аргументы
    parser.add_argument('-e', '--epoch', default="10000",
                        help='Количество эпох')
    parser.add_argument('-a', '--algorithm', default='PPO',
                        help='Название алгоритма ["evolve", "DDPG", "td3", "PPO"]')

    args = parser.parse_args()

    print(args.env, args.algorithm, args.epoch)

    # Использование аргументов
    if args.env == "1":
        env = Simulation.Environment(
            field_size=50.0
        )
    elif args.env == "2":
        env = Simulation.Environment2(
            field_size=50.0
        )
    elif args.env == "3":
        env = Simulation.Environment3(
            field_size=50.0
        )
    else:
        exit(0)

    if args.algorithm == "evolve":
        start_evolution_algorithm(env)
    elif args.algorithm == "DDPG":
        start_ddpg(env, int(args.epoch))
    elif args.algorithm == "td3":
        start_td3(env, int(args.epoch))
    elif args.algorithm == "PPO":
        start_PPO(env, int(args.epoch))
    else:
        exit(0)
