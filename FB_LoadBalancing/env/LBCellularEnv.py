import numpy as np
import gym
from gym import spaces
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import convert_action_env_to_agent, convert_to_one_hot
import timing


def _get_handover_interruption_time(old_association: np.ndarray,
                                    new_association: np.ndarray,
                                    rlf_prob: float,
                                    interruption_time_dict: Dict) -> np.ndarray:
    """
    Types of event possible:
    - success:
        - Inter-site HO
        - Intra-site HO
    - RLF
    """
    ho_user_bool_idx = old_association != new_association
    ho_user_bool_idx = np.logical_or(ho_user_bool_idx[:, 0], ho_user_bool_idx[:, 1])

    if np.sum(ho_user_bool_idx) > 0:
        # This is just the mock function for the time being.
        # True if RLF, False if success
        ho_event_rlf_bool = np.random.rand(ho_user_bool_idx.shape[0]) < rlf_prob
        interruption_time = ho_event_rlf_bool * interruption_time_dict["RLF"]
        interruption_time[np.logical_not(ho_event_rlf_bool)] = interruption_time_dict["successful_ho"]
    else:
        interruption_time = np.zeros(old_association.shape[0])

    return interruption_time


def _compute_number_bs(dataframe_dict) -> int:
    """
    Computes the number of sectors in the network using the dataframe_dict
    """
    num_bs = 0
    # for ue_idx in dataframe_dict.keys():
    #    for freq_band in dataframe_dict[ue_idx].keys():
    #        df = dataframe_dict[ue_idx][freq_band]
    #        num_bs = np.max([np.max(df['serving pci']), num_bs])

    # Just for testing, let's make is 6 as it is the case in this setup for a given frequency.
    num_bs = 6
    return num_bs


class LBCellularEnv(gym.Env):
    """LBCellularEnv Environment that follows gym interface

        self.user_bs_association: numpy array of length self.NumUE. 
                                self.user_bs_association[i] = index of the BS with which 
                                                                    user i is associated.

        self.sinr_mat: 3D numpy array. SINR measurements
                    Dimension: NumUE X NumBS X NumFreq

        self.state: Look at self.get_state()
        dataframe_dict: List of dataframes representing measurements along user trajectories.
                        len(DataFrameList) == NumUE.
        
    """

    def __init__(self,
                 env_params_dict: Dict):
        super(LBCellularEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=
        (3, 3, 3), dtype=np.uint8)

        self.name = "LoadBalancingCentralized"
        self.current_step = 0

        self.NOISE_dBm_per_Hz = env_params_dict["NOISE_dBm_per_Hz"] \
                                + env_params_dict["NOISE_FIGURE"]
        self.rlf_prob = env_params_dict["RLF_PROB"]
        self.interruption_time_dict = env_params_dict["INTERRUPTION_TIME_DICT"]

        self._slot_duration_ms = env_params_dict["SLOT_DURATION_MS"]

        self.BW_Hz = env_params_dict["BW"]
        self.tx_power_correction = env_params_dict["TX_POWER_CORRECTION_dB"]
        self.interference_fraction = env_params_dict["INTERFERENCE_FRACTION"]
        self.partitioned = env_params_dict["PARTITIONED"]
        self.state = self.reset(dataframe_dict=env_params_dict["DATAFRAME_DICT"])

    def _take_action(self,
                     action: np.ndarray):
        """
        Apply changes to the environment based on the action.
        Update the current user-BS association matrix
        """
        self.user_bs_association = action.copy()
        self._update_state()

        return

    def get_user_loc_mat(self):
        """
            Get the user locs for the next time step from Quadriga generated csv files.
        """

        loc_mat = np.zeros((self.NumUE, 2))

        for ue_idx in range(self.NumUE):
            df = self._DataFrameDict[ue_idx][self.freq_bands[0]]

            if self.partitioned:
                effective_step = (np.floor(self.current_step) / len(df)).astype(int)
                if self.effective_step >= len(df):
                    breakpoint()
                x_array = df.iloc[effective_step, 0]
                y_array = df.iloc[effective_step, 1]
                loc_mat[ue_idx, 0] = x_array
                loc_mat[ue_idx, 1] = y_array

            else:
                if self.current_step >= len(df):
                    breakpoint()
                x_array = df.iloc[self.current_step, 0]
                y_array = df.iloc[self.current_step, 1]
                loc_mat[ue_idx, 0] = x_array
                loc_mat[ue_idx, 1] = y_array

        return loc_mat

    def _next_rsrp_mat(self) -> np.ndarray:
        """
            Get the RSRP measurements for the next time step from Quadriga generated csv files.
        """

        rsrp_mat = np.zeros((self.NumUE, self.NumBS, self.NumFreq))

        # These have been manually set. They are column indices from RSRP csv files generated by Quadriga
        pci_idx_list = [4, 5, 6, 7, 8, 9]
        rsrp_idx_list = [10, 11, 12, 13, 14, 15]

        for ue_idx in range(self.NumUE):
            for freq_idx, freq_band in enumerate(self.freq_bands):
                df = self._DataFrameDict[ue_idx][freq_band]

                if self.partitioned:
                    effective_step = (np.floor(self.current_step) / len(df)).astype(int)
                    if effective_step >= len(df):
                        breakpoint()
                    pci_array = df.iloc[effective_step, pci_idx_list].to_numpy().astype(int)
                    pci_array = pci_array - 1
                    rsrp_array = df.iloc[effective_step, rsrp_idx_list].to_numpy()
                    rsrp_mat[ue_idx, pci_array, freq_idx] = rsrp_array
                    
                else:
                    if self.current_step >= len(df):
                        breakpoint()
                    pci_array = df.iloc[self.current_step, pci_idx_list].to_numpy().astype(int)
                    pci_array = pci_array - 1
                    rsrp_array = df.iloc[self.current_step, rsrp_idx_list].to_numpy()
                    rsrp_mat[ue_idx, pci_array, freq_idx] = rsrp_array

        return rsrp_mat

    def _rsrp_to_sinr(self, rsrp_mat_db):
        """
        Convert the input rsrp_mat (NumUE, NumBS, NumFreq) to
        SINR matrix (NumUE, NumBS, NumFreq) .

        The input rsrp should be in dB
        """

        rsrp_mat = 10 ** (rsrp_mat_db / 10)

        if len(rsrp_mat.shape) == 2:
            rsrp_mat = np.expand_dims(rsrp_mat, axis=2)

        sum_rsrp = np.sum(rsrp_mat, axis=1)
        sum_rsrp = np.expand_dims(sum_rsrp, axis=1)
        sum_rsrp = np.tile(sum_rsrp, (1, self.NumBS, 1))

        interference = (sum_rsrp - rsrp_mat) * self.interference_fraction
        noise = 10 ** (self.NOISE_dBm_per_Hz / 10) * self.BW_mat_Hz
        sinr_mat = rsrp_mat / (noise + interference)

        return np.squeeze(sinr_mat)

    def _update_dataframe_dict(self, dataframe_dict):
        """
        Update self._DataFrameDict. Can be used for switching out the scenarios for different episodes.
        """
        self._DataFrameDict = dataframe_dict
        self.NumBS = _compute_number_bs(dataframe_dict)
        self.NumUE = len(dataframe_dict)
        # print(len(dataframe_dict), self.current_step)
        self.freq_bands = list(dataframe_dict[0].keys())
        self.NumFreq = len(self.freq_bands)
        
        if self.partitioned:
            self.max_episode_length = len(dataframe_dict[0][self.freq_bands[0]]) * self.NumUE
        else:
            self.max_episode_length = len(dataframe_dict[0][self.freq_bands[0]])

        return

    def _update_sinr_measurements(self):
        """
        Sample the next state (from pandas dataframe?) based on the current time.
        Process the RSRP matrix from the dataframe to SINR matrix.
        """
        next_rsrp_mat_db = self._next_rsrp_mat() + self.tx_power_correction
        next_sinr_mat = self._rsrp_to_sinr(next_rsrp_mat_db)
        self.sinr_mat = next_sinr_mat
        self.rsrp_db_mat = next_rsrp_mat_db
        self._update_state()

        return

    def _update_state(self):
        """
            Return the state required by the agent.
            env_state: 3D matrix of dimension (NumUE, NumBS, NumFreq)
            Agent state is a concatenation of SINR measurements and load measurements
        """
        # Change state from (NumUE, NumBS, NumFreq) to (NumUE, NumBS * NumFreq)
        sinr_mat = np.concatenate([self.sinr_mat[:, :, 0], self.sinr_mat[:, :, 1]], axis=1)

        # Change user_association from (NumUE, 2) to (NumUE, 1) where
        # user_association[i] = bs_idx + freq_idx * NumBS.
        association = convert_action_env_to_agent(action_env=self.user_bs_association,
                                                  num_bs=self.NumBS).astype(int)
        association = convert_to_one_hot(input_array=association,
                                         n_values=self.NumBS * self.NumFreq)

        load_per_bs = np.sum(association, axis=0)
        load_per_bs = np.tile(load_per_bs, (self.NumUE, 1))

        # stack sinr_mat and load per base station into a 3D matrix
        self.state = np.stack([sinr_mat, load_per_bs], axis=2)

        return

    def _get_reward(self,
                    action: np.ndarray) -> np.ndarray:
        """
        Use the current user_bs_association and action (the new user_bs_association)
        to generate the HO interruption times and the sum utility.
        """

        interruption_time_array \
            = _get_handover_interruption_time(old_association=self.user_bs_association,
                                              new_association=action,
                                              rlf_prob=self.rlf_prob,
                                              interruption_time_dict=self.interruption_time_dict)
        se_matrix = np.log2(1 + self.sinr_mat[:, :, :])
        rate_matrix = self.BW_mat_Hz * se_matrix

        per_user_utility = np.zeros(self.NumUE)

        for it in range(self.NumUE):
            bs_idx = action[it, 0]
            freq_idx = action[it, 1]
            rate = rate_matrix[it, bs_idx, freq_idx]
            rate = (1 - (interruption_time_array[it] / self._slot_duration_ms)) * rate
            # print(rate, (1 - (interruption_time_array[it] / self._slot_duration_ms)),
            #       rate/(1 - (interruption_time_array[it] / self._slot_duration_ms)))
            # breakpoint()

            load = action == (bs_idx, freq_idx)
            load = np.sum(np.logical_and(load[:, 0], load[:, 1]))

            eff_rate = rate / load
            per_user_utility[it] = np.log(eff_rate + 1e-7)
            # if (interruption_time_array[it] / self._slot_duration_ms) < 0.7:
            #     per_user_utility[it] -= 100

        return per_user_utility

    def step(self, action):
        # Execute one time step within the environment

        self.current_step += 1

        reward = self._get_reward(action)
        self._update_sinr_measurements()
        obs = self.get_state()
        done = self.current_step >= self.max_episode_length - 1

        self._take_action(action)

        return obs, reward, done, {}

    def get_state(self):
        """
            Return the state required by the agent.
        """

        return self.state.copy()

    def get_capacity_tensor(self):
        """
        Returns the 3D capacity tensor
        """

        return self.BW_mat_Hz * np.log2(1 + self.sinr_mat)

    def get_sinr_tensor(self):
        """
        Returns the 3D SINR tensor
        """

        return self.sinr_mat.copy()

    def get_rsrp_tensor(self):
        """
        Returns the 3D RSRP tensor, measured in dB
        """

        return self.rsrp_db_mat.copy()

    def get_slot_duration(self):

        return self._slot_duration_ms

    def reset(self, dataframe_dict=None):
        # Reset the state of the environment to an initial state

        self.current_step = 0
        if dataframe_dict is not None:
            self._update_dataframe_dict(dataframe_dict)

        assert self.NumFreq == self.BW_Hz.shape[0], "incorrect initialization \
                                                                   of BW_Hz and NumFreq"

        self.BW_mat_Hz = np.tile(np.reshape(self.BW_Hz, (1, 1, -1)),
                                 (self.NumUE, self.NumBS, 1))

        self.user_bs_association = np.zeros((self.NumUE, 2))
        self._update_sinr_measurements()

        return self.get_state()

    def close(self):
        pass

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        # The render function is needed by the gym interface, must never be called.

        pass
