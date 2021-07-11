# This file defines some utilities to train the DQN
# this is mostly a copy-paste from:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 

import sys
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('DQN')
from Replay_Memory import Transition

# select_action - will select an action accordingly to an epsilon greedy policy.
# Simply put, we’ll sometimes use our model for choosing the action, and sometimes we’ll just sample one uniformly.
# The probability of choosing a random action will start at EPS_START and will decay exponentially towards EPS_END.
# EPS_DECAY controls the rate of the decay. 

def select_action(args, arguments, states):
    if args.distributed:
        if args.global_rank == 0:
            sample = random.random()
        else:
            sample = None

        sample = arguments['comm_world'].bcast(sample, root=0)
    else:
        sample = random.random()

    eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
        math.exp(-1. * arguments['epoch'] / args.eps_decay)
    # arguments['steps_done'] += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            outputs_x, outputs_y = arguments['policy_dqn_model'](states)
            fixation_pos_y = outputs_y.max(1)[1]
            fixation_pos_y = fixation_pos_y.float()/args.num_of_actions
            fixation_pos_x = outputs_x.max(1)[1]
            fixation_pos_x = fixation_pos_x.float()/args.num_of_actions
            fixation = [fixation_pos_x, fixation_pos_y]
            return torch.transpose(torch.stack(fixation), 0, 1)

    else:
        return torch.rand((args.batch_size,2))
        # return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

# select_random_action - will sample an action uniformly (at random).
def select_random_action(args):
    return torch.rand((args.batch_size,2))

# select_action_from_policy - will use our model for choosing the action.
def select_action_from_policy(args, arguments, states):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        outputs_x, outputs_y = arguments['policy_dqn_model'](states)
        fixation_pos_y = outputs_y.max(1)[1]
        fixation_pos_y = fixation_pos_y.float()/args.num_of_actions
        fixation_pos_x = outputs_x.max(1)[1]
        fixation_pos_x = fixation_pos_x.float()/args.num_of_actions
        fixation = [fixation_pos_x, fixation_pos_y]
        return torch.transpose(torch.stack(fixation), 0, 1)









# Finally, the code for training our DQN model.

# Here, you can find an optimize_foveator function that performs a single step of the optimization.
# It first samples a batch, concatenates all the tensors into a single one, computes
# Q(s_t, a_t) and V(s_{t+1}) = \max_a Q(s_{t+1}, a), and combines them into our loss.
# We also use a target network to compute V(s_{t+1}) for added stability.
# The target network has its weights kept frozen most of the time, but is updated with the policy network’s weights every so often.
# This is usually a set number of steps but we shall use episodes for simplicity.
def optimize_foveator(args, arguments):
    assert len(arguments['dqn_memory']) >= args.batch_size

    transitions = arguments['dqn_memory'].sample(args.batch_size)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))


    state_batch = torch.cat(batch.state).view(args.batch_size, -1)
    action_batch = torch.cat(batch.action).view(args.batch_size, -1)
    next_state_batch = torch.cat(batch.next_state).view(args.batch_size, -1)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    policy_ouputs_x, policy_ouputs_y = arguments['policy_dqn_model'](state_batch)
    action_batch_x_poss = action_batch[:,0].view(args.batch_size,-1)
    action_batch_y_poss = action_batch[:,1].view(args.batch_size,-1)
    state_action_values_x_poss = policy_ouputs_x.gather(1, (action_batch_x_poss*args.num_of_actions).long())
    state_action_values_y_poss = policy_ouputs_y.gather(1, (action_batch_y_poss*args.num_of_actions).long())
    state_action_values = [state_action_values_x_poss, state_action_values_y_poss]
    state_action_values = torch.stack(state_action_values, 1).squeeze(2)
    state_action_values = torch.mean(state_action_values, 1)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    target_outputs_x, target_outputs_y = arguments['target_dqn_model'](next_state_batch)
    next_state_values_x_poss = target_outputs_x.max(1)[0]
    next_state_values_y_poss = target_outputs_y.max(1)[0]
    next_state_values = [next_state_values_x_poss, next_state_values_y_poss]
    next_state_values = torch.stack(next_state_values, 1).detach()
    next_state_values = torch.mean(next_state_values, 1)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    arguments['dqn_optimizer'].zero_grad()
    loss.backward()
    for param in arguments['policy_dqn_model'].parameters():
        param.grad.data.clamp_(-1, 1)
    arguments['dqn_optimizer'].step()



    return loss

