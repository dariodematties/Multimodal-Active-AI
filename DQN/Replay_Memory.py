# This is the Replay Memory copy pasted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import random
from collections import namedtuple, deque

# It stores the transitions that the agent observes, allowing us to reuse this data later.
# By sampling from it randomly, the transitions that build up a batch are decorrelated.
# It has been shown that this greatly stabilizes and improves the DQN training procedure.

# For this, we’re going to need two classses:

# Transition - a named tuple representing a single transition in our environment.
# It essentially maps (state, action) pairs to their (next_state, reward) result, with the state being the foveated spans of a fixation.


Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))


# ReplayMemory - a cyclic buffer of bounded size that holds the transitions observed recently.
# It also implements a .sample() method for selecting a random batch of transitions for training.

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
