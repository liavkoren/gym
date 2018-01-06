import gym
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


num_samples = 2000
theta = np.zeros((4, 1))
discount_rate = 0.9
y = np.ones((num_samples, 1))
state_set = np.zeros((num_samples, 4))


def feature_map(state):
    return state


def value(state):
    return theta.T.dot(feature_map(state))


def move(state, direction):
    env.reset()
    env.env.state = state
    state, reward, done, _ = env.step(direction)
    return state, done


def theta_grad(s):
    return (theta.dot(s.T).T - y) * s


def fitted_value_iteration():
    global theta
    sample_bounds = 0.35

    state_set = np.random.uniform(-sample_bounds, sample_bounds, size=(num_samples, 4))
    print('---')
    for index, sample in enumerate(state_set):
        y_temp = []
        for action in [0, 1]:
            state_prime, done = move(sample, action)
            diffs = np.linalg.norm(state_set - state_prime, axis=1)
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

    theta_history = []
    theta = theta.T
    for _ in range(2000):
        learning_rate = .95
        theta -= learning_rate*theta_grad(state_set).sum(axis=0)/num_samples
        theta_history.append(theta)
    theta = theta.T
    # plt.plot(theta_history)

    '''
    classifier = linear_model.LinearRegression()
    classifier.fit(state_set, y)
    return classifier
    '''


def go(solver):
    for i_episode in range(20):
        score = 0
        observation = env.reset()
        for t in range(1000):
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
    for _ in range(20):
        solver = fitted_value_iteration()
        print('-'*20)
    # print(f'Learned weights: {solver.coef_} intercept {solver.intercept_}')
    # go(solver)
    # print(f'Learned weights: {solver.coef_} intercept {solver.intercept_}')

