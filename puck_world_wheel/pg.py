import time

import gym
import torch
from puck_world.envs import AgentType

from puck_world_wheel.random_agent import RandomAgent
from puck_world_wheel.pg_agent import PolicyGradAgent, PolicyNet
from puck_world_wheel.greedy_agent import GreedyAgent


def train_chaser(env_name,
                 need_restore,
                 episode_count,
                 close=False):
    """
    Use Policy-Gradient to train runner.
    """
    env = gym.make(env_name)

    chaser = PolicyGradAgent(50,
                             50,
                             30,
                             (1, 0, 0),
                             AgentType.Chaser,
                             features_n=5,
                             actions_n=4,
                             discounted_value=0.99,
                             learning_rate=0.0001,
                             need_restore=need_restore)
    runner = RandomAgent(300,
                         300,
                         30,
                         (0, 1, 0),
                         AgentType.Runner,
                         env)

    env.add_agent(chaser)
    env.add_agent(runner)
    for epi in range(episode_count):
        chaser_state = env.reset()
        step = 0
        while True:
            if step > 2000:
                break
            env.render(close=close)
            left_action, right_action = chaser.act(chaser_state)
            chaser.set_left_v(left_action)
            chaser.set_right_v(right_action)
            chaser_state_, reward, done, _ = env.step(chaser.type, 1)
            chaser.rewards.append(reward)
            chaser_state = chaser_state_
            step += 1
            if done:
                print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                break
            else:
                env.render(close=close)
                runner_action = runner.act()
                runner_action, _, done, _ = env.step(runner.type, runner_action)
                step += 1
                if done:
                    print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                    break
        # 每个回合结束更新模型
        chaser.optimize_model()
        if epi % 100 == 0:
            chaser.save()


def greedy_chase(env, episode_count, close=False):
    env = gym.make(env)
    net = PolicyNet(5, 4)
    chaser = GreedyAgent(50,
                         50,
                         30,
                         (1, 0, 0),
                         AgentType.Chaser,
                         features_n=5,
                         actions_n=4,
                         net=net,
                         load_path='model/pg.pkl')
    runner = RandomAgent(300,
                         300,
                         30,
                         (0, 1, 0),
                         AgentType.Runner,
                         env)
    env.add_agent(chaser)
    env.add_agent(runner)
    for epi in range(episode_count):
        chaser_state = env.reset()
        step = 0
        while True:
            env.render(close=close)
            left_action, right_action = chaser.act(chaser_state)
            chaser.set_left_v(left_action)
            chaser.set_right_v(right_action)
            chaser_state, _, done, _ = env.step(chaser.type, 1)
            step += 1
            if done:
                print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                break
            else:
                runner_action = runner.act()
                runner_action, _, done, _ = env.step(runner.type, runner_action)
                env.render(close=close)
                step += 1
                if done:
                    print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                    break


if __name__ == '__main__':
    env = 'puck-world-wheel-v0'
    train_chaser(env, need_restore=False, episode_count=20000, close=True)
    # greedy_chase(env, episode_count=1000, close=False)
