from matplotlib import pyplot as plt
import os
import glob

import torch

from Policies.UserAssociationPolicies import *
from Policies.DRQNAgent.DRQN import Q_net
from env.LBCellularEnv import LBCellularEnv
from utils import load_ue_trajectories, convert_action_agent_to_env
import timing


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def get_handover_count(old_association, new_association):
    """
    This function assumes association is 2D matrix of dimensions (NumUE, 2).
    """
    ho_count_dict = {}
    ho_bool = old_association != new_association

    ho_count_dict["inter_site_ho_count"] = np.sum(ho_bool[:, 0])
    ho_count_dict['inter_freq_ho_count'] = np.sum(ho_bool[:, 1])
    ho_count_dict["intra_site_ho_count"] = np.sum(np.logical_and(np.logical_not(ho_bool[:, 0]),
                                                                 ho_bool[:, 1]))
    ho_count_dict["total_ho_count"] = np.sum(np.logical_or(ho_bool[:, 0], ho_bool[:, 1]))

    return ho_count_dict


def evaluate_rl_policy(data_path_list, model_path, model_name, environment):
    """
    Evaluate a trained RL model on the given data set.
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    q_net = Q_net(state_space=environment.NumBS * environment.NumFreq,
                  action_space=environment.NumBS * environment.NumFreq).to(device)
    load_model(model=q_net, path=os.path.join(model_path, model_name))
    q_net.eval()

    print("Evaluating RL policy")
    reward_list_per_episode = []
    action_list_per_episode = []
    ho_count_per_episode = []
    rate_per_episode = []
    for episode_num in range(num_episodes):
        df_dict = load_ue_trajectories(path=data_path_list[np.random.randint(low=0, high=len(data_path_list))])
        obs = environment.reset(dataframe_dict=df_dict)
        done = False

        h, c = q_net.init_hidden_state(batch_size=environment.NumUE)
        episode_reward_list = []
        episode_action_list = []
        episode_ho_count_list = []
        episode_rate_list = []
        while not done:
            # Get action
            obs_torch = torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0)
            q_values, h, c = q_net.sample_q_value(obs_torch, h.to(device), c.to(device), epsilon_test)
            a = np.argmax(q_values, axis=1)

            # the sampled action a is an index out of the range(env.NumBS * env.NumFreq)
            # action_env is different. Look at the env doc
            action_env = convert_action_agent_to_env(action_agent=a,
                                                     num_bs=environment.NumBS)
            if len(episode_action_list) > 0:
                ho_count_dict = get_handover_count(old_association=episode_action_list[-1],
                                                   new_association=action_env)
                episode_ho_count_list.append(ho_count_dict)
            episode_action_list.append(action_env)
            # Do action
            s_prime, r, done, _ = environment.step(action_env)
            obs_prime = s_prime

            obs = obs_prime

            episode_reward_list.append(np.sum(r))
            episode_rate_list.append(np.exp(r))
        # Log the reward
        reward_list_per_episode.append(episode_reward_list)
        action_list_per_episode.append(episode_action_list)
        ho_count_per_episode.append(episode_ho_count_list)
        rate_per_episode.append(episode_rate_list)

    return reward_list_per_episode, action_list_per_episode, ho_count_per_episode, rate_per_episode


def evaluate_convex_program(data_path_list, environment):
    """
    Evaluate the performance of the convex program given data set
    """
    reward_list_per_episode = []
    action_list_per_episode = []
    ho_count_per_episode = []
    rate_per_episode = []
    print("Evaluating convex policy")
    for episode_num in range(num_episodes):
        df_dict = load_ue_trajectories(path=data_path_list[np.random.randint(low=0, high=len(data_path_list))])
        obs = environment.reset(dataframe_dict=df_dict)
        done = False

        episode_reward_list = []
        episode_action_list = []
        episode_ho_count_list = []
        episode_rate_list = []
        while not done:
            # Get action
            action_env = convex_program_association(capacity_matrix=environment.get_capacity_tensor())

            if len(episode_action_list) > 0:
                ho_count_dict = get_handover_count(old_association=episode_action_list[-1],
                                                   new_association=action_env)
                episode_ho_count_list.append(ho_count_dict)
            episode_action_list.append(action_env)
            # Do action
            s_prime, r, done, _ = environment.step(action_env)
            obs_prime = s_prime

            obs = obs_prime

            episode_reward_list.append(np.sum(r))
            episode_rate_list.append(np.exp(r))
        # Log the reward
        reward_list_per_episode.append(episode_reward_list)
        action_list_per_episode.append(episode_action_list)
        ho_count_per_episode.append(episode_ho_count_list)
        rate_per_episode.append(episode_rate_list)

    return reward_list_per_episode, action_list_per_episode, ho_count_per_episode,rate_per_episode


def evaluate_max_sinr_policy(data_path_list, environment):
    """
        Evaluate the performance of the max sinr policy given data set
    """
    reward_list_per_episode = []
    action_list_per_episode = []
    ho_count_per_episode = []
    rate_per_episode = []
    for episode_num in range(num_episodes):
        df_dict = load_ue_trajectories(path=data_path_list[np.random.randint(low=0, high=len(data_path_list))])
        obs = environment.reset(dataframe_dict=df_dict)
        done = False

        episode_reward_list = []
        episode_action_list = []
        episode_ho_count_list = []
        episode_rate_list = []
        while not done:
            # Get action
            action_env = max_sinr_association(sinr_matrix=environment.get_sinr_tensor())

            if len(episode_action_list) > 0:
                ho_count_dict = get_handover_count(old_association=episode_action_list[-1],
                                                   new_association=action_env)
                episode_ho_count_list.append(ho_count_dict)
            episode_action_list.append(action_env)
            # Do action
            s_prime, r, done, _ = environment.step(action_env)
            obs_prime = s_prime

            obs = obs_prime

            episode_reward_list.append(np.sum(r))
            episode_rate_list.append(np.exp(r))
        # Log the reward
        reward_list_per_episode.append(episode_reward_list)
        action_list_per_episode.append(episode_action_list)
        ho_count_per_episode.append(episode_ho_count_list)
        rate_per_episode.append(episode_rate_list)
    return reward_list_per_episode, action_list_per_episode, ho_count_per_episode, rate_per_episode


def evaluate_max_rate_policy(data_path_list, environment):
    """
        Evaluate the performance of the max rate policy given data set
    """
    reward_list_per_episode = []
    action_list_per_episode = []
    ho_count_per_episode = []
    rate_per_episode = []
    for episode_num in range(num_episodes):
        df_dict = load_ue_trajectories(path=data_path_list[np.random.randint(low=0, high=len(data_path_list))])
        obs = environment.reset(dataframe_dict=df_dict)
        done = False

        episode_reward_list = []
        episode_action_list = []
        episode_ho_count_list = []
        episode_rate_list = []
        while not done:
            # Get action
            action_env = max_rate_association(rate_matrix=environment.get_capacity_tensor())

            if len(episode_action_list) > 0:
                ho_count_dict = get_handover_count(old_association=episode_action_list[-1],
                                                   new_association=action_env)
                episode_ho_count_list.append(ho_count_dict)
            episode_action_list.append(action_env)
            # Do action
            s_prime, r, done, _ = environment.step(action_env)
            obs_prime = s_prime

            obs = obs_prime

            episode_reward_list.append(np.sum(r))
            episode_rate_list.append(np.exp(r))
        # Log the reward
        reward_list_per_episode.append(episode_reward_list)
        action_list_per_episode.append(episode_action_list)
        ho_count_per_episode.append(episode_ho_count_list)
        rate_per_episode.append(episode_rate_list)
    return reward_list_per_episode, action_list_per_episode, ho_count_per_episode, rate_per_episode


def evaluate_max_rsrp_policy(data_path_list, environment):
    """
        Evaluate the performance of the max rsrp policy given data set
    """
    reward_list_per_episode = []
    action_list_per_episode = []
    ho_count_per_episode = []
    rate_per_episode = []
    for episode_num in range(num_episodes):
        df_dict = load_ue_trajectories(path=data_path_list[np.random.randint(low=0, high=len(data_path_list))])
        obs = environment.reset(dataframe_dict=df_dict)
        done = False

        episode_reward_list = []
        episode_action_list = []
        episode_ho_count_list = []
        episode_rate_list = []
        while not done:
            # Get action
            action_env = max_rsrp_association(rsrp_matrix=environment.get_rsrp_tensor())

            if len(episode_action_list) > 0:
                ho_count_dict = get_handover_count(old_association=episode_action_list[-1],
                                                   new_association=action_env)
                episode_ho_count_list.append(ho_count_dict)
            episode_action_list.append(action_env)
            # Do action
            s_prime, r, done, _ = environment.step(action_env)
            obs_prime = s_prime

            obs = obs_prime

            episode_reward_list.append(np.sum(r))
            episode_rate_list.append(np.exp(r))
        # Log the reward
        reward_list_per_episode.append(episode_reward_list)
        action_list_per_episode.append(episode_action_list)
        ho_count_per_episode.append(episode_ho_count_list)
        rate_per_episode.append(episode_rate_list)
    return reward_list_per_episode, action_list_per_episode, ho_count_per_episode, rate_per_episode


if __name__ == "__main__":

    save_results_flag = True
    epsilon_test = 1e-16
    num_episodes = 20
    np.random.seed(412)
    # import UE trajectories into DataFrames
    # path = '/Users/mg57437/Documents/Manan Backup/lecture notes/GradSchool/RA/' \
    #       + "LoadBalancing/FB_Quadriga/savedResults/Scenario 0.7/urban_drive_5eNB_zapdos/trial*"

    path = '/Users/rl33442/Documents/UT Courses/Year 1/Semester 2/Reinforcement Learning/RL-Project-LB/UE_CSV_Files-10x1Antenna/*'
    path_list = glob.glob(path)

    DF_dict = load_ue_trajectories(path=path_list[1])

    # Set gym environment
    env_params = {
        "DATAFRAME_DICT": DF_dict,
        "RLF_PROB": 0.2,
        "INTERRUPTION_TIME_DICT": {"RLF": 90.79,
                                   "successful_ho": 20},
        "BW": np.array([10e6, 60e6]),
        "TX_POWER_CORRECTION_dB": -10,
        "INTERFERENCE_FRACTION": 1,
        "NOISE_dBm_per_Hz": -174,
        "NOISE_FIGURE": 10,
        "SLOT_DURATION_MS": 100,
        "PARTITIONED": False
    }

    test_environment = LBCellularEnv(env_params_dict=env_params)

    # Create agent
    # test_model_path = 'ArticunoFiles/LoadBalancingCentralized-run4'
    test_model_path = 'old_runs/runs_10x1_dqn_lr1e-3'
    test_model_name = 'LB_DRQN_centralized.pth'

    reward_list_per_episode_rl, action_list_per_episode_rl, ho_count_per_episode_rl, rate_per_episode_rl \
         = evaluate_rl_policy(data_path_list=path_list, model_path=test_model_path, model_name=test_model_name,
                              environment=test_environment)
    timing.log("Done Evaluating RL policy")
    if save_results_flag:
        np.save(os.path.join(test_model_path, "Evaluated_reward"), np.array(reward_list_per_episode_rl))
        np.save(os.path.join(test_model_path, "Evaluated_action"), np.array(action_list_per_episode_rl))
        np.save(os.path.join(test_model_path, "Evaluated_ho_count_dict"), ho_count_per_episode_rl)
        np.save(os.path.join(test_model_path, "Evaluated_rate"), np.array(rate_per_episode_rl))
    #
    # reward_list_per_episode_convex_program, \
    #     action_list_per_episode_convex_program, \
    #     ho_count_per_episode_convex_program, \
    #     rate_per_episode_convex_program \
    #     = evaluate_convex_program(data_path_list=path_list, environment=test_environment)
    # timing.log("Done evaluating convex policy")
    # if save_results_flag:
    #     np.save(os.path.join(test_model_path, "Evaluated_reward_convex_program"),
    #             np.array(reward_list_per_episode_convex_program))
    #     np.save(os.path.join(test_model_path, "Evaluated_action_convex_program"),
    #             np.array(action_list_per_episode_convex_program))
    #     np.save(os.path.join(test_model_path, "Evaluated_ho_count_dict_convex_program"),
    #             ho_count_per_episode_convex_program)
    #     np.save(os.path.join(test_model_path, "Evaluated_rate_convex_program"),
    #             np.array(rate_per_episode_convex_program))

    print("evaluating max sinr")
    reward_list_per_episode_max_sinr, \
        action_list_per_episode_max_sinr, \
        ho_count_per_episode_max_sinr, \
        rate_per_episode_max_sinr \
        = evaluate_max_sinr_policy(data_path_list=path_list, environment=test_environment)
    timing.log("Done evaluating max sinr policy")
    if save_results_flag:
        np.save(os.path.join(test_model_path, "Evaluated_reward_max_sinr"),
                np.array(reward_list_per_episode_max_sinr))
        np.save(os.path.join(test_model_path, "Evaluated_action_max_sinr"),
                np.array(action_list_per_episode_max_sinr))
        np.save(os.path.join(test_model_path, "Evaluated_ho_count_dict_max_sinr"),
                ho_count_per_episode_max_sinr)
        np.save(os.path.join(test_model_path, "Evaluated_rate_max_sinr"),
                np.array(rate_per_episode_max_sinr))

    print("evaluating max rate")
    reward_list_per_episode_max_rate, \
        action_list_per_episode_max_rate, \
        ho_count_per_episode_max_rate, \
        rate_per_episode_max_rate \
        = evaluate_max_rate_policy(data_path_list=path_list, environment=test_environment)
    timing.log("Done evaluating max rate policy")
    if save_results_flag:
        np.save(os.path.join(test_model_path, "Evaluated_reward_max_rate"),
                np.array(reward_list_per_episode_max_rate))
        np.save(os.path.join(test_model_path, "Evaluated_action_max_rate"),
                np.array(action_list_per_episode_max_rate))
        np.save(os.path.join(test_model_path, "Evaluated_ho_count_dict_max_rate"),
                ho_count_per_episode_max_rate)
        np.save(os.path.join(test_model_path, "Evaluated_rate_max_rate"),
                np.array(rate_per_episode_max_rate))

    print("evaluating max rsrp")
    reward_list_per_episode_max_rsrp, \
        action_list_per_episode_max_rsrp, \
        ho_count_per_episode_max_rsrp, \
        rate_per_episode_max_rsrp \
        = evaluate_max_rsrp_policy(data_path_list=path_list, environment=test_environment)
    timing.log("Done evaluating max rsrp policy")
    if save_results_flag:
        np.save(os.path.join(test_model_path, "Evaluated_reward_max_rsrp"),
                np.array(reward_list_per_episode_max_rsrp))
        np.save(os.path.join(test_model_path, "Evaluated_action_max_rsrp"),
                np.array(action_list_per_episode_max_rsrp))
        np.save(os.path.join(test_model_path, "Evaluated_ho_count_dict_max_rsrp"),
                ho_count_per_episode_max_rsrp)
        np.save(os.path.join(test_model_path, "Evaluated_rate_max_rsrp"),
                np.array(rate_per_episode_max_rsrp))

    # breakpoint()
