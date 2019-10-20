import time

import gym
import torch

from e_greedy_agent import EGreedyAgent
from greedy_agent import GreedyAgent
from grid_world.envs import AgentType
from random_agent import RandomAgent
from utils import get_reverse_action


def chaser_dqn(env_name, episode_count=1000, display=True, fps=10):
    env = gym.make(env_name)
    chaser = EGreedyAgent(default_reward=-1.0,
                          name='chaser',
                          color=(1.0, 0.0, 0.0),
                          env=env,
                          agent_type=AgentType.Chaser,
                          features_n=4,
                          memory_capacity=500)

    runner = RandomAgent(default_reward=1.0,
                         name='runner',
                         color=(0.0, 1.0, 0.0),
                         env=env,
                         agent_type=AgentType.Runner)

    env.add_agent(chaser)
    env.add_agent(runner)

    reward = 0
    done = False

    total_steps = 0

    for epi in range(episode_count):
        state_map = env.reset()
        chaser_info = state_map[chaser.name]
        runner_info = state_map[runner.name]

        chaser_x = chaser_info['state'][0]
        chaser_y = chaser_info['state'][1]
        runner_x = runner_info['state'][0]
        runner_y = runner_info['state'][1]

        chaser_state = torch.FloatTensor([[chaser_x, chaser_y, runner_x, runner_y]])

        runner_state = torch.FloatTensor([[runner_x, runner_y, chaser_x, chaser_y]])
        step = 0
        while True:
            if display:
                env.render()
                time.sleep(1 / fps)

            action = chaser.act(chaser_state)
            chaser_poi, direction, reward, done, _ = env.step(action.item(), chaser.name)
            chaser_x, chaser_y = chaser_poi[0], chaser_poi[1]
            chaser_state_ = torch.FloatTensor([[chaser_x, chaser_y, runner_x, runner_y]])

            reward = torch.tensor([reward], dtype=torch.float)
            chaser.memory.push(chaser_state, action, chaser_state_, reward)

            chaser_state = chaser_state_

            chaser.optimize_model()
            step += 1
            total_steps += 1

            if done:
                print('Episode: %d\tsteps: %d\t' % (epi + 1, step + 1))
                break
            else:
                runner_action = runner.act(runner_state)
                runner_poi, runner_dir, _, done, _ = env.step(runner_action.item(), runner.name)
                if display:
                    env.render()
                    time.sleep(1 / fps)
                runner_x, runner_y = runner_poi[0], runner_poi[1]
                runner_state = [runner_x, runner_y, chaser_x, chaser_y]
                step += 1
                if done:
                    # 添加 正值 reward 到 集合中
                    a = get_reverse_action(runner_action)
                    s = chaser_state
                    r = 10
                    r = torch.tensor([r], dtype=torch.float)
                    s_ = torch.FloatTensor([[chaser_x, chaser_y, runner_x, runner_y]])
                    chaser.memory.push(s, a, s_, r)
                    print('Episode: %d\tsteps: %d\t' % (epi + 1, step + 1))
                    break
        # Update the target network, copying all weights and biases in DQN
        if epi % 10 == 0:
            # target net params replaced
            print('Target net params Replaced!')
            chaser.target_net.load_state_dict(chaser.policy_net.state_dict())
    chaser.save('chaser-1000.pkl')

def greedy_chaser(env_name, episode_count, load_path, display=True, fps=10):
    env = gym.make(env_name)
    chaser = GreedyAgent(default_reward=-1.0,
                         name='chaser',
                         color=(1.0, 0.0, 0.0),
                         env=env,
                         agent_type=AgentType.Chaser,
                         load_path=load_path,
                         features_n=4)

    runner = RandomAgent(default_reward=1.0,
                         name='runner',
                         color=(0.0, 1.0, 0.0),
                         env=env,
                         agent_type=AgentType.Runner)

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

        chaser_state = torch.FloatTensor([[chaser_x, chaser_y, runner_x, runner_y]])

        runner_state = torch.FloatTensor([[runner_x, runner_y, chaser_x, chaser_y]])

        step = 0
        while True:
            if display:
                env.render()
                time.sleep(1 / fps)

            action = chaser.act(chaser_state)
            chaser_poi, direction, _, done, _ = env.step(action, chaser.name)
            chaser_x, chaser_y = chaser_poi[0], chaser_poi[1]
            chaser_state_ = torch.FloatTensor([[chaser_x, chaser_y, runner_x, runner_y]])

            chaser_state = chaser_state_
            step += 1

            if done:
                print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                break
            else:
                runner_action = runner.act(runner_state)
                runner_poi, runner_dir, _, done, _ = env.step(runner_action, runner.name)
                if display:
                    env.render()
                    time.sleep(1 / fps)
                runner_x, runner_y = runner_poi[0], runner_poi[1]
                runner_state = torch.FloatTensor([[runner_x, runner_y, chaser_x, chaser_y]])

                step += 1
                if done:
                    print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                    break


if __name__ == '__main__':
    env = 'multi-agent-8x8-v0'
    chaser_dqn(env, episode_count=10000, display=False, fps=10)
    # greedy_chaser(env, episode_count=1000, load_path='./model/chaser-1000.pkl')