import numpy as np
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        old_V = np.array(V, copy=True)
        for s in range(env.nS):
            v = 0
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += policy[s, a] * prob * (reward + discount_factor * V[next_state])
            V[s] = v

        if np.sum((V - old_V) ** 2) < theta:
            break
        old_V = np.array(V, copy=True)

    return np.array(V)


random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
print(np.floor(v.reshape(env.shape)))
