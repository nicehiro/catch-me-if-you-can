import time

import gym

from e_greedy_agent import EGreedyAgent
from greedy_agent import GreedyAgent
from grid_world.envs import single_agent_env


def chaser_dqn(env_name, episode_counts, display=False, fps=10):
    env = gym.make(env_name)
    chaser = EGreedyAgent(features_n=2,
                          actions_n=4,
                          n_width=3,
                          n_height=4)
    epi = 0
    for epi in range(episode_counts):
        state = env.reset()
        steps = 0

        if epi != 0 and epi % 100 == 0:
            chaser.save()
        if epi % 10 == 0:
            print(chaser.Q)

        while True:
            if display:
                time.sleep(1 / fps)
                env.render()
            action = chaser.act(state)
            state_, reward, done, _ = env.step(action)
            chaser.update(state, action, reward, state_)
            state = state_
            steps += 1
            if done:
                print('Episode: %d\tSteps: %d' % (epi + 1, steps))
                break


def chaser_greedy(env_name, episode_counts, display=True, fps=10):
    env = gym.make(env_name)
    chaser = GreedyAgent(actions_n=4)
    epi = 0
    for epi in range(episode_counts):
        state = env.reset()
        steps = 0
        while True:
            if display:
                time.sleep(1 / fps)
                env.render()
            action = chaser.act(state)
            state_, _, done, _ = env.step(action)
            steps += 1
            state = state_
            if done:
                print('Episode: %d\tSteps: %d' % (epi + 1, steps))
                break


if __name__ == '__main__':
    env = 'movan-world-v0'
    # chaser_dqn(env, episode_counts=2000, display=False, fps=10)
    chaser_greedy(env, episode_counts=100)
