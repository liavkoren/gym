from math import radians


import gym
import numpy as np

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
    values = {n: 0 for n in range(-1, 162)}
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
    starting_state = env.reset()
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
                    print(f'Episode finished after {t+1} time steps with a best score of {best_score}.')
                break


def state2state_number(state):
    one_degree = 0.0174532
    six_degrees = 0.1047192
    twelve_degrees = 0.2094384
    fifty_degrees = 0.87266
    x, x_dot, theta, theta_dot = state

    if (not -2.4 < x < 2.4) or (not -twelve_degrees < theta < twelve_degrees):
        return -1

    box = 0

    if x < -0.8:
        box = 0
    elif x < 0.8:
        box = 1
    else:
        box = 2

    if x_dot < -0.5:
        pass
    elif x_dot < 0.5:
        box += 3
    else:
        box += 6

    if theta < -six_degrees:
        pass
    elif theta < -one_degree:
        box += 9
    elif theta < 0:
        box += 18
    elif theta < one_degree:
        box += 27
    elif theta < six_degrees:
        box += 36
    else:
        box += 45

    if theta_dot < -fifty_degrees:
        pass
    elif theta_dot < fifty_degrees:
        box += 54
    else:
        box += 108

    return box


def get_state_map():
    quantized_state_map = {}
    twelve_degrees = 0.2094384
    twelve_degrees = 0.2094384
    fifty_degrees = 0.87266
    print('hi!')
    while len(quantized_state_map) < 162:
        x = 4.8 * np.random.random() - 2.4
        x_dot = 2 * np.random.random() - 1
        theta = 2*twelve_degrees * np.random.random() - twelve_degrees
        theta_dot = 4 * fifty_degrees * np.random.random() - 2 * fifty_degrees
        random_state = (x, x_dot, theta, theta_dot)
        quant = state2state_number(random_state)
        if quant in quantized_state_map:
            continue
        quantized_state_map[quant] = random_state
    quantized_state_map[-1] = (-10, -10, -10, -10)
    return quantized_state_map


state_map = None


def dequantize_state(state_number):
    global state_map
    if state_map is None:
        state_map = get_state_map()

    return state_map[state_number]


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    buckets = [10, 10, 10, 10]

    low = env.observation_space.low
    high = env.observation_space.high
    low[1] = -0.5
    low[3] = -radians(50)
    high[1] = 0.5
    high[3] = radians(50)

    quant = Quantizer(low=low, high=high, buckets=buckets)

    go(env)
