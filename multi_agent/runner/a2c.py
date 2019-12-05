import time

import gym
import torch
from grid_world.envs import AgentType

from agent.a2c_agent import A2CAgent, A2CNet
from agent.actor_critic_agent import ActorCriticNet
from agent.greedy_agent import GreedyAgent
from multi_agent.runner.greedy_run import runner_greedy
from utils import get_reverse_action


def train_runner_with_a2c(env_name,
                          runner_need_restore,
                          chaser_restore_path,
                          episode_count,
                          display=False,
                          fps=10):
    """
    Use Actor-Critic Advantage method to train runner.
    """
    env = gym.make(env_name)

    runner = A2CAgent(
        default_reward=0.5,
        name='runner',
        color=(0, 1, 0),
        env=env,
        agent_type=AgentType.Runner,
        features_n=4,
        discounted_value=0.99,
        init_value=0.0,
        learning_rate=1e-4,
        need_restore=runner_need_restore
    )

    chaser = GreedyAgent(default_reward=-1.0,
                         name='chaser',
                         color=(1.0, 0.0, 0.0),
                         env=env,
                         agent_type=AgentType.Chaser,
                         load_path=chaser_restore_path,
                         features_n=4)
    env.add_agent(chaser)
    env.add_agent(runner)
    for epi in range(episode_count):
        state_map = env.reset()
        chaser_info = state_map[chaser.name]
        runner_info = state_map[runner.name]

        chaser_x = chaser_info['state'][0]
        chaser_y = chaser_info['state'][1]
        runner_x = runner_info['state'][0]
        runner_y = runner_info['state'][1]
        chaser_state = [chaser_x, chaser_y, runner_x, runner_y]
        runner_state = [runner_x, runner_y, chaser_x, chaser_y]
        step = 0
        while True:
            if display:
                env.render()
                time.sleep(1 / fps)
            action = chaser.act(chaser_state)
            chaser_poi, direction, _, done, _ = env.step(action, chaser.name)
            chaser_x, chaser_y = chaser_poi[0], chaser_poi[1]
            chaser_state_ = [chaser_x, chaser_y, runner_x, runner_y]
            chaser_state = chaser_state_
            step += 1
            if step >= 500:
                # When episode's steps above 500, break up this episode
                break
            if done:
                # When chaser chase runner, also need to add new transition
                r = -10
                runner_action = get_reverse_action(action)
                runner_state_ = [runner_x, runner_y, chaser_x, chaser_y]
                runner_action_ = runner.act(runner_state_)
                # runner.rewards.append(r)
                # runner.states_.append(runner_state_)
                print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                break
            else:
                runner_action = runner.act(runner_state)
                runner_poi, runner_dir, reward, done, _ = env.step(
                    runner_action, runner.name)
                if display:
                    env.render()
                    time.sleep(1 / fps)
                runner_x, runner_y = runner_poi[0], runner_poi[1]
                runner_state_ = [runner_x, runner_y, chaser_x, chaser_y]
                runner_action_ = runner.act(runner_state_)
                runner.rewards.append(reward)
                runner.states_.append(runner_state_)
                runner_state = runner_state_
                step += 1
                if done:
                    print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                    break
        # 每个回合结束更新模型
        runner.optimize_model()
        if epi % 100 == 0:
            runner.save()


if __name__ == '__main__':
    env = 'multi_agent-8x8-v0'
    # train_runner_with_a2c(env,
    #                       runner_need_restore=False,
    #                       chaser_restore_path='./model/chaser-1000.pkl',
    #                       episode_count=40000,
    #                       display=False,
    #                       fps=10)
    a2c_net = A2CNet(4, 4)
    a2c_net.load_state_dict(torch.load('./model/a2c_runner.pkl'))
    a2c_net = a2c_net.probs
    runner_greedy(env,
                  episode_count=100,
                  chaser_restore_path='./model/chaser-2000.pkl',
                  display=True,
                  fps=10,
                  policy_net=a2c_net)
