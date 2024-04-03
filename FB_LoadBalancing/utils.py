import numpy as np
import torch
import os
import glob
import pandas as pd


def get_output_folder(env_name, parent_dir='output_files'):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    env_name: str; Name of the environment. Will be used for naming experiment
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)

    return parent_dir


def load_ue_trajectories(path):
    extension = "*.csv"
    search_exp = os.path.join(path, extension)
    files = glob.glob(pathname=search_exp)
    ue_idx_list = set()
    fc_list = set()

    dataframe_dict = {}

    for file in files:
        ue_idx = int(file.split('/')[-1].split('_')[1][2:]) - 1
        ue_idx_list.add(ue_idx)

        fc = float(file.split('/')[-1].split('_')[-1].split('.')[0])
        fc_list.add(fc)

        if ue_idx not in dataframe_dict.keys():
            dataframe_dict[ue_idx] = {}

        dataframe_dict[ue_idx][fc] = pd.read_csv(file)

    return dataframe_dict


def convert_to_one_hot(input_array, n_values=None):
    """
    Convert N-dimensional input_array to (N+1)-dimensional one_hot encoded array.
    if input_array[i,j,k] = 3, output_array[i,j,k,:] = one_hot(3)
    """
    if isinstance(input_array, np.ndarray):
        if n_values is None:
            n_values = np.max(input_array) + 1
        output_array = np.eye(n_values)[input_array]

        return output_array

    elif isinstance(input_array, torch.Tensor):
        if n_values is None:
            n_values = input_array.max() + 1
        output_array = torch.eye(n_values)[input_array]

        return output_array


def reshape_env2agent(env_var):
    """
    Reshape the (N,*,NumBS,NumFreq) matrix to (N,*,NumBS * NumFreq)
    The input could be of dimensions: (seq_len, batch_size, NumUE, NumBS, NumFreq) or
                                        (batch_size, NumUE, NumBS, NumFreq) or
                                        (NumUE, NumBS, NumFreq) or
                                        (NumBS, NumFreq)
    """
    NumBS = env_var.shape[-2]
    NumFreq = env_var.shape[-1]

    if isinstance(env_var, torch.Tensor):
        agent_var = env_var.detach().clone()
        while len(agent_var.shape) < 5:
            agent_var = agent_var.unsqueeze(0)

        agent_var = agent_var.permute(0, 1, 2, 4, 3)
        agent_var = agent_var.reshape(agent_var.shape[0], agent_var.shape[1],
                                      agent_var.shape[2], NumBS * NumFreq)

        return agent_var.squeeze()

    if isinstance(env_var, np.ndarray):
        agent_var = env_var.copy()
        while len(agent_var.shape) < 5:
            agent_var = np.expand_dims(agent_var, axis=0)

        agent_var = agent_var.transpose(0, 1, 2, 4, 3)
        agent_var = agent_var.reshape(agent_var.shape[0], agent_var.shape[1],
                                      agent_var.shape[2], NumBS * NumFreq)

        return np.squeeze(agent_var)


def reshape_agent2env(agent_var, NumBS, NumFreq):
    """
    Reshape the (N,*,NumBS*NumFreq) matrix to (N,*,NumBS,NumFreq)
    The input could be of dimensions: (seq_len, batch_size, NumUE, NumBS*NumFreq) or
                                        (batch_size, NumUE, NumBS*NumFreq) or
                                        (NumUE, NumBS*NumFreq) or
                                        (NumBS*NumFreq, )
    """

    if isinstance(agent_var, torch.Tensor):
        env_var = agent_var.detach().clone()
        while len(env_var.shape) < 4:
            env_var = env_var.unsqueeze(0)

        env_var = env_var.reshape(env_var.shape[0], env_var.shape[1],
                                  env_var.shape[2], NumFreq, NumBS)
        env_var = env_var.permute(0, 1, 2, 4, 3)

        return env_var.squeeze()

    if isinstance(agent_var, np.ndarray):
        env_var = agent_var.copy()
        while len(env_var.shape) < 4:
            env_var = np.expand_dims(env_var, axis=0)

        env_var = env_var.reshape(env_var.shape[0], env_var.shape[1],
                                  env_var.shape[2], NumFreq, NumBS)
        env_var = env_var.transpose(0, 1, 2, 4, 3)

        return np.squeeze(env_var)


def convert_action_agent_to_env(action_agent, num_bs):
    """
    Convert the (N,*, 1) action tensor to (N,*, 2) tensor.
    For agent action is num_users X 1. action[i] represents the index of BS and frequency band.
    For env action is num_users X 2. action[i, 0] is the index of the BS (or sector). action[i, 1] is the index of the
    frequency band.

    num_bs: Total number of physical BS available.
    num_freq: Total number of frequency bands available.
    """
    action_env = np.stack([action_agent % num_bs, action_agent // num_bs], axis=len(action_agent.shape))

    return action_env


def convert_action_env_to_agent(action_env, num_bs):
    """
    Convert the (N,*, 2) action tensor to (N,*, 1) tensor.
    For agent action is num_users X 1. action[i] represents the index of BS and frequency band.
    For env action is num_users X 2. action[i, 0] is the index of the BS (or sector). action[i, 1] is the index of the
    frequency band.

    num_bs: Total number of physical BS available.
    num_freq: Total number of frequency bands available.
    """
    action_agent = action_env[..., 0] + action_env[..., 1] * num_bs

    return action_agent


def action_env_from_q_values_agent(q_value_agent, num_bs, num_freq):
    q_values_env = reshape_agent2env(agent_var=q_value_agent, NumBS=num_bs, NumFreq=num_freq)

    action_env = np.zeros((q_values_env.shape[0], 2), dtype=int)
    for ue_idx in range(action_env.shape[0]):
        bs_idx, freq_idx = np.where(q_values_env[ue_idx, :, :] == np.max(q_values_env[ue_idx, :, :]))
        action_env[ue_idx, 0] = bs_idx[0]
        action_env[ue_idx, 1] = freq_idx[0]

    return action_env


if __name__ == "__main__":

    blah = torch.from_numpy(np.arange(24).reshape(4, 3, 2))
    idx = torch.from_numpy(np.random.randint(low=0, high=2, size=(4, 3)))
    one_hot_idx = convert_to_one_hot(idx)

    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            print(i, j)
            print(idx[i, j])
            print(blah[i, j, idx[i, j]])
            print(" ")

    breakpoint()
    print(blah[one_hot_idx.bool()])
