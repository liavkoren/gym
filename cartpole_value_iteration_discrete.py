from itertools import product
from math import radians

import gym

from quantizer import Quantizer

simulator = None
quant = Quantizer


def step(state, action):
    """
    reset
    env.env.state = state
    action
    """
    global simulator
    if simulator is None:
        simulator = gym.make('CartPole-v1')
    simulator.reset()
    simulator.env.state = state
    # <np.array(state)>, reward, done, {}
    return simulator.step(action)


def value_iteration(gamma=.95):
    """ Return a dict of tuple to value. """
    theta = 10e-4
    values = init_state_values()
    gamma = .95
    while True:
        delta = 0
        for quantized_state, old_value in values.items():
            actual_state = dequantize_state(quantized_state)
            best_reward = float('-inf')
            for action in [0, 1]:
                new_state, reward, done, _ = step(actual_state, action)
                if done:
                    reward = 0
                if reward > best_reward:
                    best_reward = reward

            values[quantized_state] = best_reward + gamma * values[state2state_number(new_state)]
            delta = max(delta, abs(old_value - values[quantized_state]))
        print(delta)
        if delta < theta:
            print(values)
            return values


def policy(env, values, state, gamma=.95):
    best_action = -1
    best_reward = float('-inf')
    for action in [0, 1]:
        new_state, reward, _, _ = step(state, action)
        expected_reward = reward + gamma * values[state2state_number(new_state)]
        if expected_reward > best_reward:
            best_reward = expected_reward
            best_action = action
    return best_action


def go(env):
    best_score = float('-inf')
    gamma = .99
    # starting_state = env.reset()
    values = value_iteration(gamma=gamma)
    for episode in range(20):
        state = env.reset()
        action = policy(env, values, state, gamma=gamma)
        score = 0
        for t in range(1000):
            env.render()
            action = policy(env, values, state, gamma=gamma)
            state, reward, done, info = env.step(action)
            score += reward
            if done:
                if score > best_score:
                    best_score = score
                    print(f'Episode finished after {t+1} time steps.')
                break


def state2state_number(state):
    global quant
    return quant.quantize(state)


def tuples():
    out = []
    global buckets
    ranges = [list(range(bucket)) for bucket in buckets]
    for tup in product(*ranges):
        out.append(tup)
    return out


def init_state_values():
    state_tuples = tuples()
    return {tup: 0 for tup in state_tuples}


state_map = None


def dequantize_state(state_tuple):
    global quant
    return quant.dequantize(state_tuple)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    """
    bucket_size = 6
    buckets = [bucket_size] * 4
    """
    buckets = [20, 6, 6, 6]
    low = env.observation_space.low
    high = env.observation_space.high
    low[1] = -0.5
    low[3] = -radians(50)
    high[1] = 0.5
    high[3] = radians(50)

    quant = Quantizer(low=low, high=high, buckets=buckets)

    go(env)
    print(buckets)
