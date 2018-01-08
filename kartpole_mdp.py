import gym
# import matplotlib.pyplot as plt
import numpy as np


num_samples = 1600
theta = np.zeros((5, 1))
discount_rate = 0.9
y = np.ones((num_samples, 1))
state_set = np.zeros((num_samples, 4))
sample_bounds = 0.35


def feature_map(state):
    return state


def value(state):
    return theta.T.dot(feature_map(state))


def move(state, direction):
    env.reset()
    env.env.state = state[:4]
    state, reward, done, _ = env.step(direction)
    return state, done


def theta_grad(s):
    return (theta.dot(s.T).T - y) * s


def fitted_value_iteration():
    global theta


    state_set = np.random.uniform(-sample_bounds, sample_bounds, size=(num_samples, 4))
    bias = np.ones((num_samples, 1))
    state_set = np.concatenate([state_set, bias], axis=1)
    for index, sample in enumerate(state_set):
        y_temp = []
        for action in [0, 1]:
            state_prime, done = move(sample, action)
            # Nearest Neigh calc:
            diffs = np.linalg.norm(state_set[:, :4] - state_prime, axis=1)
            sort_indicies = np.argsort(diffs)
            neighbors_list = state_set[sort_indicies[:3], :]
            q_action = 0
            if not done:
                for neighbor in neighbors_list:
                    q_action += discount_rate * value(neighbor)
                q_action /= 3
                q_action += 1

            if 240 <= index <= 300:
                print(f'i: {index} state: {sample} action: {action} value: {q_action}')
            y_temp.append(q_action)
        y[index] += max(y_temp)

    theta = theta.T
    num_iters = range(1000)
    for _ in num_iters:
        learning_rate = .95
        param_scale = np.linalg.norm(theta.ravel())
        update = -learning_rate * theta_grad(state_set).sum(axis=0)/num_samples
        update_scale = np.linalg.norm(update.ravel())
        theta += learning_rate*update
        print(f'update scale: {update_scale/param_scale} (should be ~10^-3)')
    theta = theta.T

    accuracy = np.mean((state_set.dot(theta) - y) < 1e-3)
    print(f'Acc: {accuracy}')

def go(solver):
    for i_episode in range(5):
        score = 0
        observation = env.reset()
        for t in range(100):
            env.render()
            action = solver.predict(observation.reshape((1, 4)))
            observation, reward, done, info = env.step(int(action[0, 0]))
            score += reward
            if done:
                print(f'Episode finished after {t+1} time steps.')
                break
        print(f'{i_episode}')


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.reset()

    for _ in range(1):
        fitted_value_iteration()

    # Make a test set:
    test_set_size = 400
    state_set = np.random.uniform(-sample_bounds, sample_bounds, size=(test_set_size, 4))
    bias = np.ones((test_set_size, 1))
    test_set = np.concatenate([state_set, bias], axis=1)
    y_test = []
