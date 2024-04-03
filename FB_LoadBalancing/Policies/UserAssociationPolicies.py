import numpy as np
import cvxpy as cp
from utils import reshape_env2agent, convert_action_agent_to_env


def convex_program_association(capacity_matrix):
    """
        Use the capacity_matrix to generate user association in 2D matrix form as required by the environment.
        num_ue = capacity_matrix.shape[0]
        num_bs = capacity_matrix.shape[1]
        num_freq = capacity_matrix.shape[2]
    """
    eps = 1e-7
    num_bs = capacity_matrix.shape[1]
    C = reshape_env2agent(capacity_matrix + eps)

    if C.shape[0] == 0:
        breakpoint()

    X = cp.Variable(C.shape, nonneg=True)
    c = cp.Parameter(C.shape, nonneg=True)
    col_sum_X = cp.sum(X, axis=0)
    row_sum_X = cp.sum(X, axis=1)

    obj = cp.sum(cp.multiply(X, cp.log(c))) + cp.sum(cp.entr(col_sum_X))

    constraints = [row_sum_X == 1,
                   X <= 1,
                   X >= 0]
    c.value = C
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.SCS)
    User_bs_assoc = np.argmax(X.value, axis=1)

    # print('Max utility acheived:', obj.value)
    return convert_action_agent_to_env(User_bs_assoc, num_bs=num_bs)


def max_sinr_association(sinr_matrix):
    """
    Return the max sinr association given the sinr_matrix
    num_ue = sinr_matrix.shape[0]
    num_bs = sinr_matrix.shape[1]
    num_freq = sinr_matrix.shape[2]
    """

    num_bs = sinr_matrix.shape[1]
    agent_sinr_mat = reshape_env2agent(sinr_matrix)
    user_bs_association = np.argmax(agent_sinr_mat, axis=1)

    return convert_action_agent_to_env(user_bs_association, num_bs=num_bs)


def max_rate_association(rate_matrix):
    """
    Return the max rate association given the rate_matrix
    num_ue = rate_matrix.shape[0]
    num_bs = rate_matrix.shape[1]
    num_freq = rate_matrix.shape[2]
    """

    num_bs = rate_matrix.shape[1]
    agent_rate_mat = reshape_env2agent(rate_matrix)
    user_bs_association = np.argmax(agent_rate_mat, axis=1)

    return convert_action_agent_to_env(user_bs_association, num_bs=num_bs)


def max_rsrp_association(rsrp_matrix):
    """
    Return the max rsrp association given the 3D rsrp_matrix
    num_ue = rsrp_matrix.shape[0]
    num_bs = rsrp_matrix.shape[1]
    num_freq = rsrp_matrix.shape[2]
    """

    num_bs = rsrp_matrix.shape[1]
    agent_rsrp_mat = reshape_env2agent(rsrp_matrix)
    user_bs_association = np.argmax(agent_rsrp_mat, axis=1)

    return convert_action_agent_to_env(user_bs_association, num_bs=num_bs)


def rand_association(state):
    user_bs_association = np.zeros(state.shape[0])

    for ue_idx in range(state.shape[0]):
        user_bs_association[ue_idx, 0] = np.random.randint(low=0, high=state.shape[1])
        user_bs_association[ue_idx, 1] = np.random.randint(low=0, high=state.shape[2])

    user_bs_association = np.hstack((np.expand_dims(user_bs_association, axis=1),
                                     np.zeros((user_bs_association.shape[0], 1))))

    return user_bs_association.astype(int)
