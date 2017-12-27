import gym
import numpy as np


def get_model(*args, **kwargs):
    return np.random.rand((4)) * 20 - 10


def go(env):
    best_score = float('-inf')
    best_model = None
    for i_episode in range(20):
        observation = env.reset()
        score = 0
        model = get_model()
        for t in range(1000):
            env.render()
            action = int(model.dot(observation) > 0)
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                if score > best_score:
                    best_model = model
                    best_score = score
                    print(f'Episode finished after {t+1} time steps with points with {best_model}.')
                break


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.reset()
    go(env)
