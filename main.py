import Simulation
import Evolution
import BasicTrain
import Visualizer
import torch
import DDPG
import TD3
import PPO
import argparse
import smartAgent


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
    ddpg = DDPG.DDPG(gamma=0.99, tau=0.001, hidden_size=512, state_size=state_size, action_size=2, batch_size=64, device="cuda", count_last_states=1, path_save_anim="DDPG/anim", path_save_model="DDPG/model")
    ddpg.train(env=env, epochs=epoch, count_steps=300)


def start_td3(env, epoch=100):
    print("start td3")
    state_size = env.get_state_size()
    td3 = TD3.TD3(gamma=0.9, tau=0.001, hidden_size=512, state_size=state_size, action_size=2, batch_size=64, device="cuda", count_last_states=1, policy_update_freq=2, policy_noise=0.3, path_save_anim="td3/anim", path_save_model="td3/model")
    # td3.load_actor("sim2_model/DDPG_10000_-10.1044.pth")
    td3.train(env=env, epochs=epoch, count_steps=300)


def save_model_run(env, path, count_steps):
    network = torch.load(path, weights_only=False)
    individual = Evolution.Individual(network, device="cuda")
    Visualizer.animation(individual=individual, env=env, device="cuda", render=True, count_steps=count_steps)


def start_new_PPOt(env, epoch=100):
    env = Simulation.Environment3(50, 0, flag_target=True, flag_nutrition=False)
    state_size = env.get_state_size()
    ppo = PPO.PPO(gamma=0.9, gae_lambda=0.95, hidden_size=64, state_size=state_size, action_size=2, batch_size=128,
                  device="cuda", count_last_states=1, trajectory_length=512, count_retrain=1,
                  path_save_anim="PPO/PPOt/anim", path_save_model="PPO/PPOt/model")

    ppo.train(env=env, epochs=epoch, count_steps=400)


def start_new_PPOe(env, epoch=100):
    env = Simulation.Environment3(50, 4, flag_target=False, flag_nutrition=False)
    state_size = env.get_state_size()
    ppo = PPO.PPO(gamma=0.9, gae_lambda=0.95, hidden_size=64, state_size=state_size, action_size=2, batch_size=128,
                  device="cuda", count_last_states=1, trajectory_length=512, count_retrain=1,
                  path_save_anim="PPO/PPOe/anim", path_save_model="PPO/PPOe/model")

    ppo.train(env=env, epochs=epoch, count_steps=400)


def start_new_PPOn(env, epoch=100):
    env = Simulation.Environment3(50, 0, flag_target=False, flag_nutrition=True)
    state_size = env.get_state_size()
    # ppo = PPO.PPO(gamma=0.99, gae_lambda=0.97, hidden_size=64, state_size=state_size, action_size=2, batch_size=64,
    #               device="cuda", count_last_states=1, trajectory_length=1024, count_retrain=10,
    #               path_save_anim="PPO/PPOn/anim", path_save_model="PPO/PPOn/model")
    ppo = PPO.SimplePPO(gamma=0.99, gae_lambda=0.95, hidden_size=32, state_size=state_size, action_size=2, batch_size=16,
                   device="cuda", trajectory_length=300, count_retrain=10,
                   path_save_anim="PPO/PPOn/anim", path_save_model="PPO/PPOn/model")
    ppo.train(env=env, epochs=epoch, eval_interval=100)


def start_DDPGn(env, epoch=100):
    env = Simulation.Environment3(50, 0, flag_target=False, flag_nutrition=True)
    state_size = env.get_state_size()
    ddpg = DDPG.DDPG(gamma=0.9, tau=0.001, hidden_size=64, state_size=state_size, action_size=2, batch_size=32,
                  device="cuda", count_last_states=1, path_save_anim="PPO/PPOn/anim", path_save_model="PPO/PPOn/model")

    ddpg.train(env=env, epochs=epoch, count_steps=400)


def start_DDPGe(env, epoch=100):
    env = Simulation.Environment3(50, 4, flag_target=False, flag_nutrition=False)
    state_size = env.get_state_size()
    ddpg = DDPG.DDPG(gamma=0.9, tau=0.001, hidden_size=64, state_size=state_size, action_size=2, batch_size=128,
                  device="cuda", count_last_states=1, path_save_anim="PPO/PPOe/anim", path_save_model="PPO/PPOe/model")

    ddpg.train(env=env, epochs=epoch, count_steps=400)


def start_DDPGt(env, epoch=100):
    env = Simulation.Environment3(50, 0, flag_target=True, flag_nutrition=False)
    state_size = env.get_state_size()
    ddpg = DDPG.DDPG(gamma=0.9, tau=0.001, hidden_size=64, state_size=state_size, action_size=2, batch_size=128,
                  device="cuda", count_last_states=1, path_save_anim="PPO/PPOt/anim", path_save_model="PPO/PPOt/model")
    ddpg.train(env=env, epochs=epoch, count_steps=400)


def start_PPO(env, epoch=100):
    print("start PPO")
    state_size = env.get_state_size()
    ppo = PPO.PPO(gamma=0.99, gae_lambda=0.95, hidden_size=512, state_size=state_size, action_size=2, batch_size=32, device="cuda", count_last_states=1, trajectory_length=4096, count_retrain=5, path_save_anim="PPO/anim", path_save_model="PPO/model")
    # td3.load_actor("sim3_2/DDPG_12900_0.4655.pth")
    ppo.train(env=env, epochs=epoch, count_steps=600)


def comb_sim3_ddpg(epoch=101):
    print("comb_sim3_DDPG")
    env = Simulation.Environment3(
        field_size=50.0, count_enemy=4, dict_reward={"enemy": 0.25,
                                                     "target": 2,
                                                     "nutrition": 1}
    )
    state_size = env.get_state_size()
    pretrained_skill = {"target":  torch.load("./pretraining_skils_sim3_ddpg/DDPGt.pth", weights_only=False),
                        "enemy":  torch.load("./pretraining_skils_sim3_ddpg/DDPGe.pth", weights_only=False),
                        "nutrition":  torch.load("./pretraining_skils_sim3_ddpg/DDPGn.pth", weights_only=False)}
    actor = smartAgent.SimpleHierarchicalActor(pretrained_skills=pretrained_skill, state_size=state_size, hidden_size=128)
    ddpg = DDPG.DDPG(gamma=0.9, tau=0.001, actor=actor, hidden_size=256, state_size=state_size, action_size=2, batch_size=64,
                     device="cuda", count_last_states=1, path_save_anim="pretraining_skils_sim3_ddpg/anim",
                     path_save_model="pretraining_skils_sim3_ddpg/model")
    ddpg.load_actor("pretraining_skils_sim3_ddpg/DDPG_15500_50.3094.pth")
    ddpg.train(env=env, epochs=epoch, count_steps=400)


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
    elif args.algorithm == "PPOe":
        start_new_PPOe(env, int(args.epoch))
    elif args.algorithm == "PPOn":
        start_new_PPOn(env, int(args.epoch))
    elif args.algorithm == "PPOt":
        start_new_PPOt(env, int(args.epoch))
    elif args.algorithm == "DDPGe":
        start_DDPGe(env, int(args.epoch))
    elif args.algorithm == "DDPGn":
        start_DDPGn(env, int(args.epoch))
    elif args.algorithm == "DDPGt":
        start_DDPGt(env, int(args.epoch))
    elif args.algorithm == "cDDPG3":
        comb_sim3_ddpg(int(args.epoch))
    else:
        exit(0)
