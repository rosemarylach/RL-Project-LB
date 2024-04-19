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
from Policies.DRQNAgent.DRQN import Q_net
from Policies.DRQNAgent.Buffers import EpisodeMemory, EpisodeBuffer
from env.LBCellularEnv import LBCellularEnv
from utils import convert_to_one_hot, convert_action_agent_to_env, get_output_folder, load_ue_trajectories

import timing


def ensure_directory_exists(directory_path):
    # This is to ensure that such a directory exists before trying to save any data.
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    raise TypeError("Not serializable")
def train(q_net=None, target_q_net=None, episode_memory=None,
          device=None,
          optimizer=None,
          batch_size=1,
          learning_rate=1e-3,
          gamma=0.99):
    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples, seq_len = episode_memory.sample()

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for i in range(batch_size):
        observations.append(samples[i]["obs"])
        actions.append(samples[i]["acts"])
        rewards.append(samples[i]["rews"])
        next_observations.append(samples[i]["next_obs"])
        dones.append(samples[i]["done"])

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)
    num_users = actions.shape[2]

    observations = torch.FloatTensor(np.transpose(observations, (1, 0, 2, 3, -1))).to(device)
    actions = torch.LongTensor(np.transpose(actions, (1, 0, 2, -1))).to(device)
    rewards = torch.FloatTensor(np.transpose(rewards, (1, 0, -1))).to(device)
    next_observations = torch.FloatTensor(np.transpose(next_observations, (1, 0, 2, 3, -1))).to(device)
    dones = torch.FloatTensor(np.transpose(dones, (1, 0))).to(device)

    h_target, c_target = target_q_net.init_hidden_state(batch_size=batch_size * num_users)

    q_target, _, _ = target_q_net(next_observations, h_target.to(device), c_target.to(device))

    q_target_max = q_target.max(-1)[0].detach()

    targets = rewards + gamma * q_target_max * dones.unsqueeze(2).repeat(1, 1, q_target_max.shape[2])
    h, c = q_net.init_hidden_state(batch_size=batch_size * num_users)
    q_out, _, _ = q_net(observations, h.to(device), c.to(device))

    q_a = q_out[actions.bool()]

    # Multiply Importance Sampling weights to loss
    loss = F.smooth_l1_loss(q_a, targets.flatten())

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

        h, c = q_net.init_hidden_state(batch_size=test_environment.NumUE)
        episode_reward_list = []
        while not done:

            # Get action
            q_values, h, c = q_net.sample_q_value(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0),
                                              h.to(device), c.to(device), epsilon_test)
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
    test_run_flag = True
    evaluate_flag = False
    load_old_model_flag = False
    # import UE trajectories into DataFrames
    # path = '/Users/mg57437/Documents/Manan Backup/lecture notes/GradSchool/RA/' \
    #        + "LoadBalancing/FB_Quadriga/savedResults/Scenario 0.7/mobility_sampled_100ms/trial*"
    # path = '/Users/mg57437/Documents/Manan Backup/lecture notes/GradSchool/RA/' \
    #        + "LoadBalancing/FB_Quadriga/savedResults/Scenario 0.7/mobility_sampled_100ms_3eNB/trial*"
    # path = '/Users/mg57437/Documents/Manan Backup/lecture notes/GradSchool/RA/' \
    #       + "LoadBalancing/FB_Quadriga/savedResults/Scenario 0.7/urban_drive/trial*"

    path = '/Users/ad53533/Documents/MATLAB/Current-Classes/Reinforcement-Learning/UE_CSV_Files/*'

    # path = '/Users/mg57437/Documents/Manan Backup/lecture notes/GradSchool/RA/' \
    #        + "LoadBalancing/FB_Quadriga/savedResults/Scenario 0.7/urban_drive_5eNB_zapdos/trial*"
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
        "RLF_PROB": 0.5,
        "INTERRUPTION_TIME_DICT": {"RLF": 90.79,
                                   "successful_ho": 20},
        "BW": np.array([10e6, 60e6]),
        "TX_POWER_CORRECTION_dB": -10,
        "INTERFERENCE_FRACTION": 0.8,
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

    # Ensure the output path exists
    ensure_directory_exists(out_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set the seed
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    # env.seed(seed)

    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/' + env_name + "_" + model_name + "_" + exp_num)

    # Set parameters
    batch_size = 16
    learning_rate = 1e-3
    buffer_len = int(100000)
    min_epi_num = 16  # Start moment to train the Q network
    episodes = 101
    print_per_iter = 20
    test_per_iter = episodes/10
    target_update_period = 4
    eps_start = 0.1
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 200

    # DRQN param
    random_update = True  # If you want to do random update instead of sequential update
    lookup_step = 10  # If you want to do random update instead of sequential update
    max_epi_len = 128
    max_epi_step = max_step
    # hidden_size = 64

    param_dict = {
        "DATA_PATH": path_list[0],
        "BATCH_SIZE": str(batch_size),
        "LEARNING_RATE": learning_rate,
        "NUM_EPISODES": str(episodes),
        "SEQ_LEN_FOR_MODEL_UPDATE": str(lookup_step),
        "NUM_BS": str(env.NumBS),
        "NUM_UE": str(env.NumUE),
        "FREQ_BANDS": list(env.freq_bands),
        "BANDWIDTHS": env_params["BW"],
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

    episode_memory = EpisodeMemory(random_update=random_update,
                                   max_epi_num=100, max_epi_len=600,
                                   batch_size=batch_size,
                                   lookup_step=lookup_step)

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

        episode_record = EpisodeBuffer()
        h, c = Q.init_hidden_state(batch_size=env.NumUE)

        t = -1
        while not done:
            t += 1
            # Get action
            q_values, h, c = Q.sample_q_value(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0),
                                              h.to(device), c.to(device), epsilon)
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

            episode_record.put([obs, a_bool_idx, r / 100.0, obs_prime, done_mask])

            obs = obs_prime

            score += np.sum(r)
            score_sum += np.sum(r)

            if len(episode_memory) >= batch_size + 1:
                train(Q, Q_target, episode_memory, device,
                      optimizer=optimizer,
                      batch_size=batch_size,
                      learning_rate=learning_rate)

                if (t + 1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- navie update
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()):  # <- soft update
                        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        print("Episode:", i)

        episode_memory.put(episode_record)

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
    # When dumping to JSON, use this function to check and convert
    with open(os.path.join(out_path, 'parameters.json'), 'w') as fp:
        json.dump(param_dict, fp, default=convert_numpy, indent=4)

    if evaluate_flag:
        np.save(os.path.join(out_path, "Reward_arrray_evaluation_during_training.npy"), np.array(test_reward_list))
    print("Results saved to ", out_path)
    env.close()
