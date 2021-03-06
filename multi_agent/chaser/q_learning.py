import time

import gym
from grid_world.envs import AgentType

from agent.multi_q_table_agent import QTEGreedyAgent
from agent.multi_q_table_greedy_agent import QTGreedyAgent
from agent.random_agent import RandomAgent


def chaser_q_learning(env_name,
                      episode_count=1000,
                      display=True,
                      fps=10,
                      need_reload=True,
                      reload_path=None):
    env = gym.make(env_name)
    chaser = QTEGreedyAgent(default_reward=-0.1,
                            name='chaser',
                            color=(1.0, 0.0, 0.0),
                            env=env,
                            agent_type=AgentType.Chaser,
                            n_width=env.n_width,
                            n_height=env.n_height,
                            need_reload=need_reload,
                            reload_path=reload_path)
    runner = RandomAgent(default_reward=1.0,
                         name='runner',
                         color=(0.0, 1.0, 0.0),
                         env=env,
                         agent_type=AgentType.Runner)
    env.add_agent(chaser)
    env.add_agent(runner)
    # Total steps contains every episode's steps
    total_steps = 0
    for epi in range(episode_count):
        state_map = env.reset()
        chaser_info = state_map[chaser.name]
        runner_info = state_map[runner.name]
        chaser_x = chaser_info['state'][0]
        chaser_y = chaser_info['state'][1]
        runner_x = runner_info['state'][0]
        runner_y = runner_info['state'][1]

        # State contains four elements
        chaser_state = [chaser_x, chaser_y, runner_x, runner_y]
        runner_state = [runner_x, runner_y, chaser_x, chaser_y]
        # current episode's steps
        step = 0
        while True:
            if display:
                env.render()
                time.sleep(1 / fps)
            action = chaser.act(chaser_state)
            chaser_poi, direction, reward, done, _ = env.step(
                action, chaser.name)
            chaser_x, chaser_y = chaser_poi[0], chaser_poi[1]
            chaser_state_ = [chaser_x, chaser_y, runner_x, runner_y]

            chaser.update(chaser_state, action, reward, chaser_state_)
            # Update current state
            chaser_state = chaser_state_
            step += 1
            total_steps += 1
            if done:
                print('Episode: %d\tsteps: %d' %
                      (epi + 1, step + 1))
                break
            else:
                runner_action = runner.act(runner_state)
                runner_poi, runner_dir, _, done, _ = env.step(
                    runner_action, runner.name)
                if display:
                    env.render()
                    time.sleep(1 / fps)
                runner_x, runner_y = runner_poi[0], runner_poi[1]
                runner_state = [runner_x, runner_y, chaser_x, chaser_y]
                step += 1
                if done:
                    # 添加 正值 reward 到 集合中
                    # a = get_reverse_action(runner_action)
                    # s = chaser_state
                    # r = 10
                    # s_ = [chaser_x, chaser_y, runner_x, runner_y]
                    # chaser.memory.push(s, a, s_, r)
                    print('Episode: %d\tsteps: %d' %
                          (epi + 1, step + 1))
                    break
            # Update the target network, copying all weights and biases in DQN
            if total_steps % 500 == 0:
                chaser.save()


def chaser_qt_greedy(env_name,
                     restore_path,
                     episode_count=1000,
                     display=True,
                     fps=10):
    env = gym.make(env_name)
    chaser = QTGreedyAgent(default_reward=-1.0,
                           name='chaser',
                           color=(1.0, 0.0, 0.0),
                           env=env,
                           agent_type=AgentType.Chaser,
                           restore_path=restore_path,
                           n_width=env.n_width,
                           n_height=env.n_height)

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
            if done:
                print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                break
            else:
                runner_action = runner.act(runner_state)
                runner_poi, runner_dir, _, done, _ = env.step(
                    runner_action, runner.name)
                if display:
                    env.render()
                    time.sleep(1 / fps)
                runner_x, runner_y = runner_poi[0], runner_poi[1]
                runner_state = [runner_x, runner_y, chaser_x, chaser_y]
                step += 1
                if done:
                    print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                    break


if __name__ == '__main__':
    env = 'multi-agent-8x8-v0'
    # chaser_q_learning(env, display=False, episode_count=40000, need_reload=False)
    chaser_qt_greedy(env, restore_path='../../model/chaser_single.npy')
