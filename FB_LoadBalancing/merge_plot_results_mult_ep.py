import numpy as np
import matplotlib.pyplot as plt
import os
# from utils import compute_long_term_throughput
import scipy.stats as stats


def merge_data(path_list, policy):
    """
        policy: "rl", "convex_program", "max_sinr", "max_rate", "max_rsrp"
    """
    path = path_list[0]
    if policy == "rl":
        train_rewards, eval_train_rewd, rewards, actions, inter_site_ho_count, inter_freq_ho_count, \
        intra_site_ho_count, total_ho_count, rate = load_results_data(path=path, policy=policy)
        for path2 in path_list[1:]:
            train_rewards2, eval_train_rewd2, rewards2, actions2, inter_site_ho_count2, inter_freq_ho_count2, \
            intra_site_ho_count2, total_ho_count2, rate2 = load_results_data(path=path2, policy=policy)
            train_rewards = np.concatenate([train_rewards, train_rewards2], axis=0)
            eval_train_rewd = np.concatenate([eval_train_rewd, eval_train_rewd2], axis=0)
            rewards = np.concatenate([rewards, rewards2], axis=0)
            actions = np.concatenate([actions, actions2], axis=0)
            inter_site_ho_count = np.concatenate([inter_site_ho_count, inter_site_ho_count2], axis=0)
            inter_freq_ho_count = np.concatenate([inter_freq_ho_count, inter_freq_ho_count2], axis=0)
            intra_site_ho_count = np.concatenate([intra_site_ho_count, intra_site_ho_count2], axis=0)
            total_ho_count = np.concatenate([total_ho_count, total_ho_count2], axis=0)
            rate = np.concatenate([rate, rate2], axis=0)

        return train_rewards, eval_train_rewd, rewards, actions, inter_site_ho_count, \
               inter_freq_ho_count, intra_site_ho_count, total_ho_count, rate

    else:
        rewards, actions, inter_site_ho_count, inter_freq_ho_count, \
        intra_site_ho_count, total_ho_count, rate = load_results_data(path=path, policy=policy)

        for path2 in path_list[1:]:
            rewards2, actions2, inter_site_ho_count2, inter_freq_ho_count2, \
            intra_site_ho_count2, total_ho_count2, rate2 = load_results_data(path=path2, policy=policy)

            rewards = np.concatenate([rewards, rewards2], axis=0)
            actions = np.concatenate([actions, actions2], axis=0)
            inter_site_ho_count = np.concatenate([inter_site_ho_count, inter_site_ho_count2], axis=0)
            inter_freq_ho_count = np.concatenate([inter_freq_ho_count, inter_freq_ho_count2], axis=0)
            intra_site_ho_count = np.concatenate([intra_site_ho_count, intra_site_ho_count2], axis=0)
            total_ho_count = np.concatenate([total_ho_count, total_ho_count2], axis=0)
            rate = np.concatenate([rate, rate2], axis=0)

        return rewards, actions, inter_site_ho_count, \
               inter_freq_ho_count, intra_site_ho_count, total_ho_count, rate


def load_results_data(path, policy):
    """
    policy: "rl", "convex_program", "max_sinr", "max_rate", "max_rsrp"
    """
    if policy == "rl":
        train_reward_array = np.load(os.path.join(path, "Train_reward_list.npy"), allow_pickle=True)
        eval_reward_training = np.zeros((10,10,10))
        # eval_reward_training = np.load(os.path.join(path,
        #                                             "Reward_arrray_evaluation_during_training.npy"),
        #                                allow_pickle=True)
        rewards = np.load(os.path.join(path, "Evaluated_reward.npy"), allow_pickle=True)
        actions = np.load(os.path.join(path, "Evaluated_action.npy"), allow_pickle=True)
        ho_count = np.load(os.path.join(path, "Evaluated_ho_count_dict.npy"), allow_pickle=True)
        rate = np.load(os.path.join(path, "Evaluated_rate.npy"), allow_pickle=True)
    else:
        rewards = np.load(os.path.join(path, "Evaluated_reward_" + policy + ".npy"), allow_pickle=True)
        actions = np.load(os.path.join(path, "Evaluated_action_" + policy + ".npy"), allow_pickle=True)
        ho_count = np.load(os.path.join(path, "Evaluated_ho_count_dict_" + policy + ".npy"), allow_pickle=True)
        rate = np.load(os.path.join(path, "Evaluated_rate_" + policy + ".npy"), allow_pickle=True)

    inter_site_ho_count = np.zeros_like(ho_count)
    inter_freq_ho_count = np.zeros_like(ho_count)
    intra_site_ho_count = np.zeros_like(ho_count)
    total_ho_count = np.zeros_like(ho_count)

    for it1 in range(ho_count.shape[0]):
        for it2 in range(ho_count.shape[1]):
            inter_site_ho_count[it1, it2] = ho_count[it1, it2]['inter_site_ho_count']
            inter_freq_ho_count[it1, it2] = ho_count[it1, it2]['inter_freq_ho_count']
            intra_site_ho_count[it1, it2] = ho_count[it1, it2]['intra_site_ho_count']
            total_ho_count[it1, it2] = ho_count[it1, it2]['total_ho_count']

    if policy == "rl":
        return train_reward_array, eval_reward_training, rewards, actions, inter_site_ho_count, \
               inter_freq_ho_count, intra_site_ho_count, total_ho_count, rate
    else:
        return rewards, actions, inter_site_ho_count, \
               inter_freq_ho_count, intra_site_ho_count, total_ho_count, rate


if __name__ == "__main__":
    path_list = ['old_runs/runs_2x2_partitioned_lr1e-3']
    path_list_newtrajs = ['old_runs/runs_2x2_dqn_lr1e-3']
    # path_list = ['output_files/LoadBalancingCentralized-run25']
    # path_list = ['ZapdosFiles/LoadBalancingCentralized-run3']
    # path_list = ['ArticunoFiles/tmp2/LoadBalancingCentralized-run4',
    #              'ArticunoFiles/tmp/LoadBalancingCentralized-run4',
    #              'ArticunoFiles/tmp3/LoadBalancingCentralized-run4']
    num_runs = 20
    num_runs_sinr_rsrp = 20
    plot_rl_flag = True
    plot_convex_program_flag = False
    plot_max_sinr_flag = True
    plot_max_rate_flag = False
    plot_max_rsrp_flag = True
    plot_rl_newtrajs_flag = True
    plot_max_rsrp_newtrajs_flag = False
    plot_max_sinr_newtrajs_flag = False
    fontsize = 12
    linewidth = 3
    smoothing_window = 10
    max_rl_util = 1

    fig_rewd, ax_rewd = plt.subplots(nrows=2, ncols=1, sharex=True)
    plt.figure("util_cdf")
    plt.figure("ho_cdf")
    plt.figure("rate_cdf")

    if plot_rl_newtrajs_flag:
        train_rewards, eval_train_rewd, rewards_rl, actions_rl, inter_site_ho_count_rl, \
        inter_freq_ho_count_rl, intra_site_ho_count_rl, total_ho_count_rl, rate_rl = merge_data(path_list_newtrajs, "rl")

        # plot reward during training
        fig, ax = plt.subplots(nrows=2, ncols=1)
        mean_reward_train = np.mean(np.sum(eval_train_rewd, axis=2), axis=1)
        std_reward_train = np.std(np.sum(eval_train_rewd, axis=2), axis=1)

        ax[0].errorbar(np.arange(mean_reward_train.shape[0]), mean_reward_train,
                       yerr=std_reward_train,
                       label='RL agent, DQN')
        ax[0].set_ylabel('Network sum utility')
        ax[0].set_xlabel('Time step')
        ax[0].legend()

        ax[1].plot(np.arange(train_rewards.shape[0]), train_rewards)

        # plot the reward during testing
        mean_reward_rl = np.mean(rewards_rl, axis=0)
        cumulative_mean_reward_rl = np.cumsum(mean_reward_rl)
        std_rewards_rl = np.std(rewards_rl, axis=0)

        # mean_reward_convex_program = np.mean(rewards_convex_program, axis=0)
        # cumulative_mean_reward_convex_program = np.cumsum(mean_reward_convex_program)
        # std_rewards_convex_program = np.std(rewards_convex_program, axis=0)
        #
        plt.figure()
        plt.semilogy(np.arange(rewards_rl.shape[1]),
                     cumulative_mean_reward_rl, label="RL agent")
        # plt.semilogy(np.arange(rewards_convex_program.shape[1]),
        #              cumulative_mean_reward_convex_program, label="Convex program")
        plt.grid(True)
        plt.ylabel('Cumulative network sum utility')
        plt.xlabel('Time step')
        plt.legend()

        mean_reward_rl = np.mean(rewards_rl, axis=0)
        cumulative_mean_reward_rl = np.cumsum(mean_reward_rl)
        std_rewards_rl = np.std(rewards_rl, axis=0)

        ax_rewd[0].errorbar(np.arange(rewards_rl.shape[1]), mean_reward_rl,
                            yerr=std_rewards_rl,
                            label='RL agent, DQN')
        ax_rewd[1].semilogy(np.arange(rewards_rl.shape[1]),
                            cumulative_mean_reward_rl, label="RL agent, DQN")

        plt.figure("util_cdf")
        util_list_rl = np.sum(rewards_rl, axis=1)
        max_rl_util = np.max(util_list_rl)
        util_list_rl = util_list_rl / max_rl_util
        plt.plot(np.sort(util_list_rl),
                 np.arange(len(util_list_rl)) / len(util_list_rl), '-',
                 linewidth=linewidth,
                 label='RL Agent, DQN')

        plt.figure("ho_cdf")
        plt.plot(np.sort(total_ho_count_rl.flatten()),
                 np.arange(len(total_ho_count_rl.flatten())) / len(total_ho_count_rl.flatten()), '-',
                 linewidth=linewidth,
                 label='RL Agent, DQN', color='blue')
        x1_RL = np.sort(np.min(np.sort(total_ho_count_rl, axis=1), axis=0))  # First line x-values

        x2_RL = np.sort(np.max(np.sort(total_ho_count_rl, axis=1),
                               axis=0))  # Second line x-values, could be slightly different from x1

        y1_RL = np.arange(len(np.sort(np.min(np.sort(total_ho_count_rl, axis=1), axis=0)))) / len(
            np.sort(np.min(np.sort(total_ho_count_rl.reshape(num_runs_sinr_rsrp, 898), axis=1), axis=0)))
        y2_RL = np.arange(len(np.sort(np.max(np.sort(total_ho_count_rl, axis=1), axis=0)))) / len(
            np.sort(np.max(np.sort(total_ho_count_rl.reshape(num_runs_sinr_rsrp, 898), axis=1), axis=0)))
        # Interpolate y2 to create a new set of y-values that correspond to x1
        y2_interpolated = np.interp(x1_RL.astype('float'), x2_RL.astype('float'), y2_RL)
        y1_interpolated = np.interp(x1_RL.astype('float'), x1_RL.astype('float'), y1_RL)

        plt.fill_between(x1_RL.astype('float'),
                         y1_interpolated.astype('float'),
                         y2_interpolated.astype('float'), color='blue', alpha=0.25)

        plt.figure("rate_cdf")
        # plt.plot(np.sort(np.min(np.sort(rate_rl.reshape(50,899*120), axis=1), axis=0)),
        #         np.arange(len(np.sort(np.min(np.sort(rate_rl.reshape(50,899*120), axis=1), axis=0)))) / len(np.sort(np.min(np.sort(rate_rl.reshape(50,899*120), axis=1), axis=0))), '-', linewidth=linewidth,
        #         label='RL Agent, reward shaping min')

        # plt.plot(np.sort(np.max(np.sort(rate_rl.reshape(50,899*120), axis=1), axis=0)),
        #         np.arange(len(np.sort(np.max(np.sort(rate_rl.reshape(50,899*120), axis=1), axis=0)))) / len(np.sort(np.max(np.sort(rate_rl.reshape(50,899*120), axis=1), axis=0))), '-', linewidth=linewidth,
        #         label='RL Agent, reward shaping max')

        x1_RL = np.sort(np.min(np.sort(rate_rl.reshape(num_runs_sinr_rsrp, 899 * 120), axis=1), axis=0))  # First line x-values

        x2_RL = np.sort(np.max(np.sort(rate_rl.reshape(num_runs_sinr_rsrp, 899 * 120), axis=1),
                               axis=0))  # Second line x-values, could be slightly different from x1

        y1_RL = np.arange(len(np.sort(np.min(np.sort(rate_rl.reshape(num_runs_sinr_rsrp, 899 * 120), axis=1), axis=0)))) / len(
            np.sort(np.min(np.sort(rate_rl.reshape(num_runs_sinr_rsrp, 899 * 120), axis=1), axis=0)))
        y2_RL = np.arange(len(np.sort(np.max(np.sort(rate_rl.reshape(num_runs_sinr_rsrp, 899 * 120), axis=1), axis=0)))) / len(
            np.sort(np.max(np.sort(rate_rl.reshape(num_runs_sinr_rsrp, 899 * 120), axis=1), axis=0)))
        # Interpolate y2 to create a new set of y-values that correspond to x1
        y2_interpolated = np.interp(x1_RL, x2_RL, y2_RL)

        plt.fill_between(x1_RL,
                         y1_RL,
                         y2_interpolated, color='blue', alpha=0.5)
        plt.plot(np.sort(rate_rl.flatten()),
                 np.arange(len(rate_rl.flatten())) / len(rate_rl.flatten()), '-', linewidth=linewidth,
                 label='RL Agent, DQN', color='blue')

    if plot_max_rsrp_newtrajs_flag:
            rewards_max_rsrp, actions_max_rsrp, \
                inter_site_ho_count_max_rsrp, inter_freq_ho_count_max_rsrp, \
                intra_site_ho_count_max_rsrp, total_ho_count_max_rsrp, \
                rate_max_rsrp = merge_data(path_list_newtrajs, policy="max_rsrp")

            mean_reward_max_rsrp = np.mean(rewards_max_rsrp, axis=0)
            cumulative_mean_reward_max_rsrp = np.cumsum(mean_reward_max_rsrp)
            std_rewards_max_rsrp = np.std(rewards_max_rsrp, axis=0)

            ax_rewd[0].errorbar(np.arange(rewards_max_rsrp.shape[1]),
                                mean_reward_max_rsrp,
                                yerr=std_rewards_max_rsrp,
                                label="Max RSRP")
            ax_rewd[1].semilogy(np.arange(rewards_max_rsrp.shape[1]),
                                cumulative_mean_reward_max_rsrp, label="Max RSRP")

            plt.figure("util_cdf")
            util_list_rsrp = np.sum(rewards_max_rsrp, axis=1)
            util_list_rsrp = util_list_rsrp / max_rl_util  # util_list_rsrp
            plt.plot(np.sort(util_list_rsrp),
                     np.arange(len(util_list_rsrp)) / len(util_list_rsrp), '-.',
                     linewidth=linewidth,
                     label='Max RSRP')

            plt.figure("ho_cdf")
            plt.plot(np.sort(total_ho_count_max_rsrp.flatten()),
                     np.arange(len(total_ho_count_max_rsrp.flatten())) / len(total_ho_count_max_rsrp.flatten()), '-.',
                     linewidth=linewidth, label='Max RSRP')

            plt.figure("rate_cdf")
            plt.plot(np.sort(rate_max_rsrp.flatten()),
                     np.arange(len(rate_max_rsrp.flatten())) / len(rate_max_rsrp.flatten()), '-.',
                     linewidth=linewidth, label='Max RSRP')







    if plot_convex_program_flag:
        rewards_convex_program, actions_convex_program, \
        inter_site_ho_count_convex_program, inter_freq_ho_count_convex_program, \
        intra_site_ho_count_convex_program, total_ho_count_convex_program, \
        rate_convex_program = merge_data(path_list, policy="convex_program")

        mean_reward_convex_program = np.mean(rewards_convex_program, axis=0)
        cumulative_mean_reward_convex_program = np.cumsum(mean_reward_convex_program)
        std_rewards_convex_program = np.std(rewards_convex_program, axis=0)

        ax_rewd[0].errorbar(np.arange(rewards_convex_program.shape[1]), mean_reward_convex_program,
                            yerr=std_rewards_convex_program,
                            label='Convex program')
        ax_rewd[1].semilogy(np.arange(rewards_convex_program.shape[1]),
                            cumulative_mean_reward_convex_program, label="Convex program")

        plt.figure("util_cdf")
        util_list_convex = np.sum(rewards_convex_program, axis=1)
        util_list_convex = util_list_convex / max_rl_util  # np.max(util_list_convex)
        plt.plot(np.sort(util_list_convex),
                 np.arange(len(util_list_convex)) / len(util_list_convex),
                 linewidth=linewidth,
                 label='Convex program')

        plt.figure("ho_cdf")
        plt.plot(np.sort(total_ho_count_convex_program.flatten()),
                 np.arange(len(total_ho_count_convex_program.flatten())) / len(total_ho_count_convex_program.flatten()),
                 linewidth=linewidth, label='Convex program')

        plt.figure("rate_cdf")
        plt.plot(np.sort(rate_convex_program.flatten()),
                 np.arange(len(rate_convex_program.flatten())) / len(rate_convex_program.flatten()),
                 linewidth=linewidth, label='Convex program')
        # plt.plot(np.sort(np.mean(rate_convex_program, axis=1).flatten()),
        #          np.arange(len(np.mean(rate_convex_program, axis=1).flatten())) \
        #          / len(np.mean(rate_convex_program, axis=1).flatten()),
        #          linewidth=linewidth, label='Convex program')
        # long_term_thp_convex_program = compute_long_term_throughput(rate_tensor=rate_convex_program,
        #                                                             smoothing_window=smoothing_window)
        # long_term_thp_convex_program = np.sum(np.log(long_term_thp_convex_program), axis=1)
        # plt.plot(np.sort(long_term_thp_convex_program.flatten()),
        #          np.arange(len(long_term_thp_convex_program.flatten())) /
        #          len(long_term_thp_convex_program.flatten()),
        #          linewidth=linewidth, label='Convex program')

    if plot_max_sinr_flag:
        rewards_max_sinr, actions_max_sinr, \
        inter_site_ho_count_max_sinr, inter_freq_ho_count_max_sinr, \
        intra_site_ho_count_max_sinr, total_ho_count_max_sinr, \
        rate_max_sinr = merge_data(path_list, policy="max_sinr")

        mean_reward_max_sinr = np.mean(rewards_max_sinr, axis=0)
        cumulative_mean_reward_max_sinr = np.cumsum(mean_reward_max_sinr)
        std_rewards_max_sinr = np.std(rewards_max_sinr, axis=0)

        ax_rewd[0].errorbar(np.arange(rewards_max_sinr.shape[1]),
                            mean_reward_max_sinr,
                            yerr=std_rewards_max_sinr,
                            label="Max SINR")
        ax_rewd[1].semilogy(np.arange(rewards_max_sinr.shape[1]),
                            cumulative_mean_reward_max_sinr, label="Max SINR")

        plt.figure("util_cdf")
        util_list_sinr = np.sum(rewards_max_sinr, axis=1)
        util_list_sinr = util_list_sinr / max_rl_util  # np.max(util_list_sinr)
        plt.plot(np.sort(util_list_sinr),
                 np.arange(len(util_list_sinr)) / len(util_list_sinr), ':',
                 linewidth=linewidth,
                 label='Max SINR')

        plt.figure("ho_cdf")
        plt.plot(np.sort(total_ho_count_max_sinr.flatten()),
                 np.arange(len(total_ho_count_max_sinr.flatten())) / len(total_ho_count_max_sinr.flatten()), ':',
                 linewidth=linewidth, label='Max SINR', color='orange')

        x1_SINR = np.sort(np.min(np.sort(total_ho_count_max_sinr, axis=1), axis=0))  # First line x-values

        x2_SINR = np.sort(np.max(np.sort(total_ho_count_max_sinr, axis=1),
                               axis=0))  # Second line x-values, could be slightly different from x1

        y1_SINR = np.arange(len(np.sort(np.min(np.sort(total_ho_count_max_sinr, axis=1), axis=0)))) / len(
            np.sort(np.min(np.sort(total_ho_count_max_sinr.reshape(num_runs_sinr_rsrp, 898), axis=1), axis=0)))
        y2_SINR = np.arange(len(np.sort(np.max(np.sort(total_ho_count_max_sinr, axis=1), axis=0)))) / len(
            np.sort(np.max(np.sort(total_ho_count_max_sinr.reshape(num_runs_sinr_rsrp, 898), axis=1), axis=0)))
        # Interpolate y2 to create a new set of y-values that correspond to x1
        y2_interpolated = np.interp(x1_SINR.astype('float'), x2_SINR.astype('float'), y2_SINR)
        y1_interpolated = np.interp(x1_SINR.astype('float'), x1_SINR.astype('float'), y1_SINR)

        plt.fill_between(x1_SINR.astype('float'),
                         y1_interpolated.astype('float'),
                         y2_interpolated.astype('float'), color='orange', alpha=0.25)

        plt.figure("rate_cdf")
        plt.plot(np.sort(rate_max_sinr.flatten()),
                 np.arange(len(rate_max_sinr.flatten())) / len(rate_max_sinr.flatten()), ':',
                 linewidth=linewidth, label='Max SINR', color='orange')
        y1_SINR = np.arange(len(np.sort(np.min(np.sort(rate_max_sinr.reshape(num_runs_sinr_rsrp,899*120), axis=1), axis=0)))) / len(np.sort(np.min(np.sort(rate_max_sinr.reshape(num_runs_sinr_rsrp,899*120), axis=1), axis=0)))
        #plt.plot(np.sort(np.min(np.sort(rate_max_sinr.reshape(50,899*120), axis=1), axis=0)),
        #         y1_SINR, '-', linewidth=linewidth,
        #         label='Max SINR, min')
        y2_SINR = np.arange(len(np.sort(np.max(np.sort(rate_max_sinr.reshape(num_runs_sinr_rsrp,899*120), axis=1), axis=0)))) / len(np.sort(np.max(np.sort(rate_max_sinr.reshape(num_runs_sinr_rsrp,899*120), axis=1), axis=0)))
        #plt.plot(np.sort(np.max(np.sort(rate_max_sinr.reshape(50,899*120), axis=1), axis=0)),
        #         y2_SINR, '-', linewidth=linewidth,
        #         label='Max SINR, max')

        # Assume x1 and y1 define the first line, and x2 and y2 define the second line.
        x1_SINR = np.sort(np.min(np.sort(rate_max_sinr.reshape(num_runs_sinr_rsrp,899*120), axis=1), axis=0))  # First line x-values

        x2_SINR = np.sort(np.max(np.sort(rate_max_sinr.reshape(num_runs_sinr_rsrp,899*120), axis=1), axis=0))  # Second line x-values, could be slightly different from x1

        # Interpolate y2 to create a new set of y-values that correspond to x1
        y2_interpolated = np.interp(x1_SINR, x2_SINR, y2_SINR)

        plt.fill_between(x1_SINR,
                         y1_SINR,
                         y2_interpolated, color='orange', alpha=0.25)

        #plt.plot(np.sort(np.mean(rate_max_sinr, axis=1).flatten()),
        #         np.arange(len(np.mean(rate_max_sinr, axis=1).flatten())) / len(
        #             np.mean(rate_max_sinr, axis=1).flatten()), ':',
        #         linewidth=linewidth, label='Max SINR')
        # long_term_thp_max_sinr = compute_long_term_throughput(rate_tensor=rate_max_sinr,
        #                                                       smoothing_window=smoothing_window)
        # long_term_thp_max_sinr = np.sum(np.log(long_term_thp_max_sinr), axis=1)
        # plt.plot(np.sort(long_term_thp_max_sinr.flatten()),
        #          np.arange(len(long_term_thp_max_sinr.flatten())) / len(long_term_thp_max_sinr.flatten()), ':',
        #          linewidth=linewidth, label='Max SINR')

    if plot_max_rate_flag:
        rewards_max_rate, actions_max_rate, \
        inter_site_ho_count_max_rate, inter_freq_ho_count_max_rate, \
        intra_site_ho_count_max_rate, total_ho_count_max_rate, \
        rate_max_rate = merge_data(path_list, policy="max_rate")

        mean_reward_max_rate = np.mean(rewards_max_rate, axis=0)
        cumulative_mean_reward_max_rate = np.cumsum(mean_reward_max_rate)
        std_rewards_max_rate = np.std(rewards_max_rate, axis=0)

        ax_rewd[0].errorbar(np.arange(rewards_max_rate.shape[1]),
                            mean_reward_max_rate,
                            yerr=std_rewards_max_rate,
                            label="Max-Rate")
        ax_rewd[1].semilogy(np.arange(rewards_max_rate.shape[1]),
                            cumulative_mean_reward_max_rate, label="Max Rate")

        plt.figure("util_cdf")
        util_list_rate = np.sum(rewards_max_rate, axis=1)
        util_list_rate = util_list_rate / max_rl_util  # np.max(util_list_sinr)
        plt.plot(np.sort(util_list_rate),
                 np.arange(len(util_list_rate)) / len(util_list_rate), ':',
                 linewidth=linewidth,
                 label='Max-Rate')

        plt.figure("ho_cdf")
        plt.plot(np.sort(total_ho_count_max_rate.flatten()),
                 np.arange(len(total_ho_count_max_rate.flatten())) / len(total_ho_count_max_rate.flatten()), '--',
                 linewidth=linewidth, label='Max-Rate')

        plt.figure("rate_cdf")
        plt.plot(np.sort(rate_max_rate.flatten()),
                 np.arange(len(rate_max_rate.flatten())) / len(rate_max_rate.flatten()), '--',
                 linewidth=linewidth, label='Max Rate')
        # plt.plot(np.sort(np.mean(rate_max_rate, axis=1).flatten()),
        #          np.arange(len(np.mean(rate_max_rate, axis=1).flatten())) / len(
        #              np.mean(rate_max_rate, axis=1).flatten()), '--',
        #          linewidth=linewidth, label='Max Rate')
        # long_term_thp_max_rate = compute_long_term_throughput(rate_tensor=rate_max_rate,
        #                                                       smoothing_window=smoothing_window)
        # long_term_thp_max_rate = np.sum(np.log(long_term_thp_max_rate), axis=1)
        # plt.plot(np.sort(long_term_thp_max_rate.flatten()),
        #          np.arange(len(long_term_thp_max_rate.flatten())) / len(long_term_thp_max_rate.flatten()), '--',
        #          linewidth=linewidth, label='Max-Rate')
    if plot_max_rsrp_flag:
        rewards_max_rsrp, actions_max_rsrp, \
        inter_site_ho_count_max_rsrp, inter_freq_ho_count_max_rsrp, \
        intra_site_ho_count_max_rsrp, total_ho_count_max_rsrp, \
        rate_max_rsrp = merge_data(path_list, policy="max_rsrp")

        mean_reward_max_rsrp = np.mean(rewards_max_rsrp, axis=0)
        cumulative_mean_reward_max_rsrp = np.cumsum(mean_reward_max_rsrp)
        std_rewards_max_rsrp = np.std(rewards_max_rsrp, axis=0)

        ax_rewd[0].errorbar(np.arange(rewards_max_rsrp.shape[1]),
                            mean_reward_max_rsrp,
                            yerr=std_rewards_max_rsrp,
                            label="Max RSRP")
        ax_rewd[1].semilogy(np.arange(rewards_max_rsrp.shape[1]),
                            cumulative_mean_reward_max_rsrp, label="Max RSRP")

        plt.figure("util_cdf")
        util_list_rsrp = np.sum(rewards_max_rsrp, axis=1)
        util_list_rsrp = util_list_rsrp / max_rl_util  # util_list_rsrp
        plt.plot(np.sort(util_list_rsrp),
                 np.arange(len(util_list_rsrp)) / len(util_list_rsrp), '-.',
                 linewidth=linewidth,
                 label='Max RSRP')

        plt.figure("ho_cdf")
        plt.plot(np.sort(total_ho_count_max_rsrp.flatten()),
                 np.arange(len(total_ho_count_max_rsrp.flatten())) / len(total_ho_count_max_rsrp.flatten()), '-.',
                 linewidth=linewidth, label='Max RSRP', color='green')

        x1_RSRP = np.sort(np.min(np.sort(total_ho_count_max_rsrp, axis=1), axis=0))  # First line x-values

        x2_RSRP = np.sort(np.max(np.sort(total_ho_count_max_rsrp, axis=1),
                               axis=0))  # Second line x-values, could be slightly different from x1

        y1_RSRP = np.arange(len(np.sort(np.min(np.sort(total_ho_count_max_rsrp, axis=1), axis=0)))) / len(
            np.sort(np.min(np.sort(total_ho_count_max_rsrp.reshape(num_runs_sinr_rsrp, 898), axis=1), axis=0)))
        y2_RSRP = np.arange(len(np.sort(np.max(np.sort(total_ho_count_max_rsrp, axis=1), axis=0)))) / len(
            np.sort(np.max(np.sort(total_ho_count_max_rsrp.reshape(num_runs_sinr_rsrp, 898), axis=1), axis=0)))
        # Interpolate y2 to create a new set of y-values that correspond to x1
        y2_interpolated = np.interp(x1_RSRP.astype('float'), x2_RSRP.astype('float'), y2_RSRP)
        y1_interpolated = np.interp(x1_RSRP.astype('float'), x1_RSRP.astype('float'), y1_RSRP)

        plt.fill_between(x1_RSRP.astype('float'),
                         y1_interpolated.astype('float'),
                         y2_interpolated.astype('float'), color='green', alpha=0.25)

        plt.figure("rate_cdf")
        plt.plot(np.sort(rate_max_rsrp.flatten()),
                 np.arange(len(rate_max_rsrp.flatten())) / len(rate_max_rsrp.flatten()), '-.',
                 linewidth=linewidth, label='Max RSRP', color='green')

        # Assume x1 and y1 define the first line, and x2 and y2 define the second line.
        x1_RSRP = np.sort(np.min(np.sort(rate_max_rsrp.reshape(num_runs_sinr_rsrp, 899 * 120), axis=1), axis=0))  # First line x-values

        x2_RSRP = np.sort(np.max(np.sort(rate_max_rsrp.reshape(num_runs_sinr_rsrp, 899 * 120), axis=1),
                                 axis=0))  # Second line x-values, could be slightly different from x1
        y1_RSRP = np.arange(len(np.sort(np.min(np.sort(rate_max_rsrp.reshape(num_runs_sinr_rsrp, 899 * 120), axis=1), axis=0)))) / len(
            np.sort(np.min(np.sort(rate_max_rsrp.reshape(num_runs_sinr_rsrp, 899 * 120), axis=1), axis=0)))
        y2_RSRP = np.arange(len(np.sort(np.max(np.sort(rate_max_rsrp.reshape(num_runs_sinr_rsrp, 899 * 120), axis=1), axis=0)))) / len(
            np.sort(np.max(np.sort(rate_max_rsrp.reshape(num_runs_sinr_rsrp, 899 * 120), axis=1), axis=0)))
        # Interpolate y2 to create a new set of y-values that correspond to x1
        # Interpolate y2 to create a new set of y-values that correspond to x1
        y1_interpolated = np.interp(x2_RSRP, x1_RSRP, y1_RSRP)

        plt.fill_between(x2_RSRP,
                         y1_interpolated,
                         y2_RSRP, color='green', alpha=0.25)
        #plt.plot(np.sort(np.mean(rate_max_rsrp, axis=1).flatten()),
        #         np.arange(len(np.mean(rate_max_rsrp, axis=1).flatten())) / len(
        #             np.mean(rate_max_rsrp, axis=1).flatten()), '-.',
        #         linewidth=linewidth, label='Max RSRP')
        # long_term_thp_max_rsrp = compute_long_term_throughput(rate_tensor=rate_max_rsrp,
        #                                                       smoothing_window=smoothing_window)
        # long_term_thp_max_rsrp = np.sum(np.log(long_term_thp_max_rsrp), axis=1)
        # plt.plot(np.sort(long_term_thp_max_rsrp.flatten()),
        #          np.arange(len(long_term_thp_max_rsrp.flatten())) / len(long_term_thp_max_rsrp.flatten()), '-.',
        #          linewidth=linewidth, label='Max RSRP')

    if plot_rl_flag:
        train_rewards, eval_train_rewd, rewards_rl, actions_rl, inter_site_ho_count_rl, \
        inter_freq_ho_count_rl, intra_site_ho_count_rl, total_ho_count_rl, rate_rl = merge_data(path_list, "rl")

        # plot reward during training
        fig, ax = plt.subplots(nrows=2, ncols=1)
        mean_reward_train = np.mean(np.sum(eval_train_rewd, axis=2), axis=1)
        std_reward_train = np.std(np.sum(eval_train_rewd, axis=2), axis=1)

        mean_rate_vec = np.mean(np.mean(rate_rl, axis=2), axis=0)

        ax[0].errorbar(np.arange(mean_reward_train.shape[0]), mean_reward_train,
                       yerr=std_reward_train,
                       label='RL agent, agent partitioning')
        ax[0].set_ylabel('Network sum utility')
        ax[0].set_xlabel('Time step')
        ax[0].legend()

        ax[1].plot(np.arange(train_rewards.shape[0]), train_rewards)

        # plot the reward during testing
        mean_reward_rl = np.mean(rewards_rl, axis=0)
        cumulative_mean_reward_rl = np.cumsum(mean_reward_rl)
        std_rewards_rl = np.std(rewards_rl, axis=0)
        #
        # mean_reward_convex_program = np.mean(rewards_convex_program, axis=0)
        # cumulative_mean_reward_convex_program = np.cumsum(mean_reward_convex_program)
        # std_rewards_convex_program = np.std(rewards_convex_program, axis=0)
        #
        plt.figure()
        plt.semilogy(np.arange(rewards_rl.shape[1]),
                     cumulative_mean_reward_rl, label="RL Agent Reward")
        plt.semilogy(np.arange(rewards_rl.shape[1]), mean_rate_vec, label="RL Agent Rate")
        # plt.semilogy(np.arange(rewards_convex_program.shape[1]),
        #              cumulative_mean_reward_convex_program, label="Convex program")
        plt.grid(True)
        plt.ylabel('Cumulative network sum utility')
        plt.xlabel('Time step')
        plt.legend()

        mean_reward_rl = np.mean(rewards_rl, axis=0)
        cumulative_mean_reward_rl = np.cumsum(mean_reward_rl)
        std_rewards_rl = np.std(rewards_rl, axis=0)

        ax_rewd[0].errorbar(np.arange(rewards_rl.shape[1]), mean_reward_rl,
                            yerr=std_rewards_rl,
                            label='RL agent,  agent partitioning')
        ax_rewd[1].semilogy(np.arange(rewards_rl.shape[1]),
                            cumulative_mean_reward_rl, label="RL agent, agent partitioning")

        """"
        mean_rate = np.mean(np.mean(rate_rl, axis=2), axis=0)
        std_rate = np.mean(np.std(rate_rl, axis=2), axis=0)
        pdf = stats.norm.pdf(np.sort(rate_rl[:,898,:].reshape(num_runs*120,1)), mean_rate[898], std_rate[898])

        plt.plot(np.sort(rate_rl[:,898,:].reshape(num_runs*120,1)), pdf)
        """

        plt.figure("util_cdf")
        util_list_rl = np.sum(rewards_rl, axis=1)
        max_rl_util = np.max(util_list_rl)
        util_list_rl = util_list_rl / max_rl_util
        plt.plot(np.sort(util_list_rl),
                 np.arange(len(util_list_rl)) / len(util_list_rl), '-',
                 linewidth=linewidth,
                 label='RL Agent, agent partitioning')

        plt.figure("ho_cdf")
        plt.plot(np.sort(total_ho_count_rl.flatten()),
                 np.arange(len(total_ho_count_rl.flatten())) / len(total_ho_count_rl.flatten()), '-',
                 linewidth=linewidth,
                 label='RL Agent, agent partitioning', color='red')
        x1_RL = np.sort(np.min(np.sort(total_ho_count_rl, axis=1), axis=0))  # First line x-values

        x2_RL = np.sort(np.max(np.sort(total_ho_count_rl, axis=1),
                               axis=0))  # Second line x-values, could be slightly different from x1

        y1_RL = np.arange(len(np.sort(np.min(np.sort(total_ho_count_rl, axis=1), axis=0)))) / len(
            np.sort(np.min(np.sort(total_ho_count_rl.reshape(num_runs, 898), axis=1), axis=0)))
        y2_RL = np.arange(len(np.sort(np.max(np.sort(total_ho_count_rl, axis=1), axis=0)))) / len(
            np.sort(np.max(np.sort(total_ho_count_rl.reshape(num_runs, 898), axis=1), axis=0)))
        # Interpolate y2 to create a new set of y-values that correspond to x1
        y2_interpolated = np.interp(x1_RL.astype('float'), x2_RL.astype('float'), y2_RL)
        y1_interpolated = np.interp(x1_RL.astype('float'), x1_RL.astype('float'), y1_RL)

        plt.fill_between(x1_RL.astype('float'),
                         y1_interpolated.astype('float'),
                         y2_interpolated.astype('float'), color='red', alpha=0.25)

        plt.figure("rate_cdf")
        #plt.plot(np.sort(np.min(np.sort(rate_rl.reshape(50,899*120), axis=1), axis=0)),
        #         np.arange(len(np.sort(np.min(np.sort(rate_rl.reshape(50,899*120), axis=1), axis=0)))) / len(np.sort(np.min(np.sort(rate_rl.reshape(50,899*120), axis=1), axis=0))), '-', linewidth=linewidth,
        #         label='RL Agent, reward shaping min')

        #plt.plot(np.sort(np.max(np.sort(rate_rl.reshape(50,899*120), axis=1), axis=0)),
        #         np.arange(len(np.sort(np.max(np.sort(rate_rl.reshape(50,899*120), axis=1), axis=0)))) / len(np.sort(np.max(np.sort(rate_rl.reshape(50,899*120), axis=1), axis=0))), '-', linewidth=linewidth,
        #         label='RL Agent, reward shaping max')

        x1_RL = np.sort(np.min(np.sort(rate_rl.reshape(num_runs,899*120), axis=1), axis=0))  # First line x-values

        x2_RL = np.sort(np.max(np.sort(rate_rl.reshape(num_runs,899*120), axis=1), axis=0))  # Second line x-values, could be slightly different from x1

        y1_RL = np.arange(len(np.sort(np.min(np.sort(rate_rl.reshape(num_runs,899*120), axis=1), axis=0)))) / len(np.sort(np.min(np.sort(rate_rl.reshape(num_runs,899*120), axis=1), axis=0)))
        y2_RL = np.arange(len(np.sort(np.max(np.sort(rate_rl.reshape(num_runs,899*120), axis=1), axis=0)))) / len(np.sort(np.max(np.sort(rate_rl.reshape(num_runs,899*120), axis=1), axis=0)))
        # Interpolate y2 to create a new set of y-values that correspond to x1
        y2_interpolated = np.interp(x1_RL, x2_RL, y2_RL)

        plt.fill_between(x1_RL,
                         y1_RL,
                         y2_interpolated, color='red', alpha=0.5)



        plt.plot(np.sort(rate_rl.flatten()),
                 np.arange(len(rate_rl.flatten())) / len(rate_rl.flatten()), '-', linewidth=linewidth,
                 label='RL Agent, agent partitioning', color='red')


        # plt.plot(np.sort(np.mean(rate_rl, axis=1).flatten()),
        #          np.arange(len(np.mean(rate_rl, axis=1).flatten())) / len(np.mean(rate_rl, axis=1).flatten()),
        #          '-', linewidth=linewidth,
        #          label='RL Agent')
        # long_term_thp_rl = compute_long_term_throughput(rate_tensor=rate_rl,
        #                                                 smoothing_window=smoothing_window)
        # long_term_thp_rl = np.sum(np.log(long_term_thp_rl), axis=1)
        # plt.plot(np.sort(long_term_thp_rl.flatten()),
        #          np.arange(len(long_term_thp_rl.flatten())) / len(long_term_thp_rl.flatten()),
        #          '-', linewidth=linewidth,
        #          label='RL Agent')

    ax_rewd[0].set_ylabel('Network sum utility')
    ax_rewd[0].set_xlabel('Time step')
    ax_rewd[0].legend()

    ax_rewd[1].set_ylabel('Cumulative network sum utility')
    ax_rewd[1].legend()

    # fig_suffix = "2eNB_BW_10_60_intf_frac_1"
    # fig_suffix = "2eNB_BW_10_60_smoothing_win_4"
    fig_suffix = "2eNB_BW_10_60"
    # Utility CDF
    plt.figure("util_cdf")
    plt.xlabel("Normalized network utility", fontsize=fontsize)
    plt.ylabel("CDF", fontsize=fontsize)
    plt.ylim(0, 1)
    plt.xlim(0.1, 1)
    # plt.xscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=fontsize)
    plt.savefig("util_cdf" + fig_suffix + ".png")

    # HO CDF
    plt.figure("ho_cdf")
    plt.xlabel("Number of handovers", fontsize=fontsize)
    plt.ylabel('CDF', fontsize=fontsize)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=fontsize)
    plt.savefig("ho_cdf" + fig_suffix + ".png")

    # rate CDF
    plt.figure("rate_cdf")
    plt.xscale('log')
    plt.xlabel("Throughput", fontsize=fontsize)
    plt.ylabel('CDF', fontsize=fontsize)
    plt.ylim(0, 1)
    # plt.xlim(1e5, 1e8)
    plt.grid(True)
    plt.legend(fontsize=fontsize)
    plt.savefig("rate_cdf" + fig_suffix + ".png")
    plt.tight_layout()

    plt.show()
    A = 1
    # breakpoint()
