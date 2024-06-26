import random
from matplotlib import pyplot as plt
import json
import os
import glob

import torch
import torch.nn.functional as F
import torch.optim as optim

# from torch.utils.tensorboard import SummaryWriter

from Policies.UserAssociationPolicies import *
from Policies.DQNAgent.DQN import Q_net
from Policies.DQNAgent.Buffers import EpisodeMemory
from env.LBCellularEnv import LBCellularEnv
from utils import convert_to_one_hot, convert_action_agent_to_env, get_output_folder, load_ue_trajectories

import timing


def train(q_net=None,
          target_q_net=None,
          memory_buffer=None,
          device=None,
          optimizer=None,
          batch_size=1,
          learning_rate=1e-3,
          gamma=0.99):
    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples = memory_buffer.sample()

    # for i in range(batch_size):
    #     observations.append(samples[i]["obs"])
    #     actions.append(samples[i]["acts"])
    #     rewards.append(samples[i]["rews"])
    #     next_observations.append(samples[i]["next_obs"])
    #     dones.append(samples[i]["done"])

    observations = np.array(samples.state)
    actions = np.array(samples.action)
    rewards = np.array(samples.reward)
    next_observations = np.array(samples.next_state)
    dones = np.array(samples.done)

    observations = torch.FloatTensor(observations).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_observations = torch.FloatTensor(next_observations).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_target = target_q_net(next_observations)

    q_target_max = q_target.max(-1)[0].detach()

    targets = rewards + gamma * q_target_max * dones.unsqueeze(1).repeat(1, q_target_max.shape[1])
    q_out = q_net(observations)
    # q_a = q_out[actions.bool()]
    q_a = q_out * actions
    q_a = q_a.sum(dim=-1)

    # Multiply Importance Sampling weights to loss
    # loss = F.smooth_l1_loss(q_a, targets.flatten())
    loss = F.smooth_l1_loss(q_a, targets)

    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate(q_net, episodes, df_dict_path_list, test_environment):
    q_net.eval()
    reward_list_per_episode = []
    epsilon_test = 1e-3
    print("Evaluating policy")
    for episode_num in range(episodes):
        df_dict = load_ue_trajectories(path=df_dict_path_list[np.random.randint(low=0,
                                                                                high=len(df_dict_path_list))])
        s = test_environment.reset(dataframe_dict=df_dict)
        obs = s
        done = False

        episode_reward_list = []
        while not done:
            # Get action
            q_values = q_net.sample_q_value(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0),
                                            epsilon_test)
            a = np.argmax(q_values, axis=1)

            # the sampled action a is an index out of the range(env.NumBS * env.NumFreq)
            # action_env is different. Look at the env doc
            action_env = convert_action_agent_to_env(action_agent=a, num_bs=env.NumBS)

            # Do action
            s_prime, r, done, _ = env.step(action_env)
            obs_prime = s_prime

            obs = obs_prime

            episode_reward_list.append(np.sum(r))

        # Log the reward
        reward_list_per_episode.append(episode_reward_list)

    q_net.train()

    return reward_list_per_episode


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_model(model, path='default.pth'):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    test_run_flag = False
    evaluate_flag = True
    load_old_model_flag = False
    # import UE trajectories into DataFrames
    # path = '/Users/mg57437/Documents/Manan Backup/lecture notes/GradSchool/RA/' \
    #        + "LoadBalancing/FB_Quadriga/savedResults/Scenario 0.7/mobility_sampled_100ms/trial*"
    # path = '/Users/mg57437/Documents/Manan Backup/lecture notes/GradSchool/RA/' \
    #        + "LoadBalancing/FB_Quadriga/savedResults/Scenario 0.7/mobility_sampled_100ms_3eNB/trial*"
    # path = '/Users/mg57437/Documents/Manan Backup/lecture notes/GradSchool/RA/' \
    #        + "LoadBalancing/FB_Quadriga/savedResults/Scenario 0.7/urban_drive/trial*"
    path = '/Users/mg57437/Documents/Manan Backup/lecture notes/GradSchool/RA/' \
           + "LoadBalancing/FB_Quadriga/savedResults/Scenario 0.7/urban_drive_zapdos/trial*"
    path_list = glob.glob(path)
    DF_dict = load_ue_trajectories(path=path_list[0])
    # for path in path_list:
    #     blah = path.split('/')[-1]
    #     DF = load_ue_trajectories(path)
    #     if len(DF) == 0:
    #         print(len(DF), blah)
    # breakpoint()
    if load_old_model_flag:
        model_path = 'output_files/LoadBalancingCentralized-run19/LB_DRQN_centralized.pth'

    # Set gym environment

    env_params = {
        "DATAFRAME_DICT": DF_dict,
        "RLF_PROB": 0.2,
        "INTERRUPTION_TIME_DICT": {"RLF": 90.79,
                                   "successful_ho": 20},
        "BW": np.array([10e6, 60e6]),
        "TX_POWER_CORRECTION_dB": 0,
        "INTERFERENCE_FRACTION": 1.0,
        "NOISE_dBm_per_Hz": -174,
        "NOISE_FIGURE": 10,
        "SLOT_DURATION_MS": 100
    }

    env = LBCellularEnv(env_params_dict=env_params)

    # Env parameters
    model_name = "LB_DRQN_centralized"
    env_name = env.name
    seed = 1
    exp_num = 'SEED' + '_' + str(seed)

    # Directory to save experiment results
    if test_run_flag:
        out_path = 'runs'
    else:
        out_path = get_output_folder(env_name=env_name)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set the seed
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    # env.seed(seed)

    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/' + env_name + "_" + model_name + "_" + exp_num)

    # Set parameters
    batch_size = 32
    learning_rate = 1e-3
    buffer_len = int(100000)
    min_epi_num = 16  # Start moment to train the Q network
    episodes = 150
    print_per_iter = 20
    test_per_iter = episodes / 10
    target_update_period = 4
    eps_start = 0.1
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 200

    # DQN param
    max_epi_len = 128
    # hidden_size = 64

    param_dict = {
        "DATA_PATH": path_list[0],
        "BATCH_SIZE": str(batch_size),
        "LEARNING_RATE": learning_rate,
        "NUM_EPISODES": str(episodes),
        "NUM_BS": str(env.NumBS),
        "NUM_UE": str(env.NumUE),
        "FREQ_BANDS": list(env.freq_bands),
        "BANDWIDTHS": list(env_params["BW"]),
        "SLOT_DURATION_MS": env_params["SLOT_DURATION_MS"],
        "INTERRUPTION_TIME_DICT": env_params["INTERRUPTION_TIME_DICT"],
        "RLF_PROB": str(env_params["RLF_PROB"])
    }
    # Create Q functions
    Q = Q_net(state_space=env.NumBS * env.NumFreq,
              action_space=env.NumBS * env.NumFreq).to(device)

    if load_old_model_flag:
        load_model(model=Q, path=model_path)

    Q_target = Q_net(state_space=env.NumBS * env.NumFreq,
                     action_space=env.NumBS * env.NumFreq).to(device)

    Q_target.load_state_dict(Q.state_dict())

    # Set optimizer
    score = 0
    score_sum = 0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    epsilon = eps_start

    episode_memory = EpisodeMemory(memory_size=100, batch_size=batch_size)

    # Train
    reward_list = []
    if evaluate_flag:
        test_reward_list = []
    timing.log("Begin Training")

    for i in range(episodes):
        # print(path_list[i % len(path_list)])
        DF_dict = load_ue_trajectories(path=path_list[i % len(path_list)])
        s = env.reset(dataframe_dict=DF_dict)
        obs = s  # Use only Position of Cart and Pole
        done = False

        t = -1
        while not done:
            t += 1
            # Get action
            q_values = Q.sample_q_value(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0),
                                        epsilon)

            a = np.argmax(q_values, axis=1)

            # the sampled action a is an index out of the range(env.NumBS * env.NumFreq)
            # action_env is different. Look at the env doc
            action_env = convert_action_agent_to_env(action_agent=a, num_bs=env.NumBS)

            # Convert the index 'a' to one hot representation.
            a_bool_idx = convert_to_one_hot(a, n_values=env.NumBS * env.NumFreq)

            # Do action
            s_prime, r, done, _ = env.step(action_env)
            obs_prime = s_prime

            # make data
            done_mask = 0.0 if done else 1.0
            episode_memory.put(state=obs, action=a_bool_idx, reward=r, next_state=obs_prime, done=done_mask)
            obs = obs_prime

            score += np.sum(r)
            score_sum += np.sum(r)

            if len(episode_memory) >= batch_size + 1:
                train(Q, Q_target, episode_memory, device, optimizer=optimizer, batch_size=batch_size,
                      learning_rate=learning_rate)

                if (t + 1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- navie update
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()):  # <- soft update
                        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        print("Episode:", i)

        epsilon = max(eps_end, epsilon * eps_decay)  # Linear annealing

        if i % print_per_iter == 0 and i != 0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                i, score_sum / print_per_iter, len(episode_memory), epsilon * 100))
            score_sum = 0.0
            save_model(Q, path=os.path.join(out_path, model_name + '.pth'))

        if evaluate_flag and (i % test_per_iter == 0):
            test_rewards = evaluate(q_net=Q, episodes=10, df_dict_path_list=path_list, test_environment=env)
            test_reward_list.append(test_rewards)
        # Log the reward

        reward_list.append(score)
        # writer.add_scalar('Rewards per episodes', score, i)
        score = 0
    timing.log("End training")
    # writer.close()
    np.save(os.path.join(out_path, "Train_reward_list.npy"), np.array(reward_list))

    with open(os.path.join(out_path, 'parameters.json'), 'w') as fp:
        json.dump(param_dict, fp, indent=4)

    if evaluate_flag:
        np.save(os.path.join(out_path, "Reward_arrray_evaluation_during_training.npy"), np.array(test_reward_list))
    print("Results saved to ", out_path)
    env.close()
