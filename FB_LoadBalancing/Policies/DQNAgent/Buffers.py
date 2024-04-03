import sys
from typing import Dict, List, Tuple

import numpy as np
import random
from collections import namedtuple, deque


class EpisodeMemory:
    """Episode memory for DQN agent"""

    def __init__(self,
                 memory_size=100,
                 batch_size=1):
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.memory = deque(maxlen=self.memory_size)
        self.transition_named_tuple = namedtuple("Transition",
                                                 ("state",
                                                  "action",
                                                  "reward",
                                                  "next_state",
                                                  "done"))

    def put(self, state, action, reward, next_state, done):
        """
        convert the inputs into a tuple and store into the buffer
        """
        transition = self.transition_named_tuple(state=state,
                                                 action=action,
                                                 reward=reward,
                                                 next_state=next_state,
                                                 done=done)
        self.memory.append(transition)

    def sample(self):
        sampled_batch = random.sample(self.memory, self.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        sampled_batch = self.transition_named_tuple(*zip(*sampled_batch))

        return sampled_batch

    def __len__(self):
        return len(self.memory)

