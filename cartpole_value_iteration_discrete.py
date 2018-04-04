from itertools import product
from math import radians
from random import random

import gym
import numpy as np

from quantizer import Quantizer


class Simulator:
    def __init__(self):
        self.engine = gym.make('CartPole-v1')

    def step(self, state, action):
        self.engine.reset()
        self.engine.env.state = state
        return self.engine.step(action)


def init_state_values(buckets):
    """ Return dict of {<state tuples>: 0}. <State tuple> is a quantized label for an observation. """
    state_tuples = []
    ranges = [list(range(bucket)) for bucket in buckets]
    for tup in product(*ranges):
        state_tuples.append(tup)
    return {tup: 0 for tup in state_tuples}


def value_iteration(quantizer, gamma=.95, theta=10e-4):
    """ Return a dict of state tuples to values. """
    values = init_state_values(quantizer.buckets)
    simulator = Simulator()
    while True:
        delta = 0
        for quantized_state, old_value in values.items():
            actual_state = quantizer.dequantize(quantized_state)
            best_reward = float('-inf')
            for action in [0, 1]:
                new_state, reward, done, _ = simulator.step(actual_state, action)
                if done:
                    reward = 0
                if reward > best_reward:
                    best_reward = reward
            values[quantized_state] = best_reward + gamma * values[quantizer.quantize(new_state)]
            delta = max(delta, abs(old_value - values[quantized_state]))
        print(f'Computing value function, delta is: {delta}', end='\r', flush=True)
        if delta < theta:
            print(values)
            return values


def policy(values, state, quantizer, simulator, gamma=.95):
    rewards = []
    for action in [0, 1]:
        new_state, reward, _, _ = simulator.step(state, action)
        expected_reward = reward + gamma * values[quantizer.quantize(new_state)]
        rewards.append(expected_reward)
    no_difference = rewards[0] == rewards[1]
    if no_difference:
        # random tie-breaking
        return random() > 0.5
    else:
        return np.argmax(rewards)


def go(env, quantizer):
    best_score = float('-inf')
    gamma = .95
    values = value_iteration(quantizer, gamma=gamma, theta=.005)
    simulator = Simulator()
    for episode in range(20):
        state = env.reset()
        action = policy(values, state, quantizer, simulator, gamma=gamma)
        score = 0
        for t in range(1000):
            env.render()
            action = policy(values, state, quantizer, simulator, gamma=gamma)
            state, reward, done, info = env.step(action)
            score += reward
            if done:
                if score > best_score:
                    best_score = score
                    print(f'Episode finished after {t+1} time steps.')
                break


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    buckets = [10, 6, 6, 6]
    low = env.observation_space.low
    low[1] = -0.5
    low[3] = -radians(50)
    high = env.observation_space.high
    high[1] = 0.5
    high[3] = radians(50)

    quantizer = Quantizer(low=low, high=high, buckets=buckets)
    go(env, quantizer)
