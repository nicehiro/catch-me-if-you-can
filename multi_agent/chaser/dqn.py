import time

import gym
from grid_world.envs import AgentType
from grid_world.envs import multi_agent_env

from agent.e_greedy_agent import EGreedyAgent
from agent.greedy_agent import GreedyAgent
from agent.random_agent import RandomAgent


def chaser_dqn(env_name,
               episode_count=1000,
               display=True,
               fps=10,
               need_reload=False,
               reload_path=None):
    """Use DQN to train Chaser to chase a random runner agent.
    """
    env = gym.make(env_name)
    # Epsilon Greedy agent with network to make policy
    chaser = EGreedyAgent(default_reward=-1.0,
                          name='chaser',
                          color=(1.0, 0.0, 0.0),
                          env=env,
                          agent_type=AgentType.Chaser,
                          features_n=4,
                          memory_capacity=1024,
                          need_reload=need_reload,
                          reload_path=reload_path)

    # Randomly make choice
    runner = RandomAgent(default_reward=1.0,
                         name='runner',
                         color=(0.0, 1.0, 0.0),
                         env=env,
                         agent_type=AgentType.Runner)

    env.add_agent(chaser)
    env.add_agent(runner)

    reward = 0
    done = False

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

            # Put this transition into memory
            chaser.memory.push(chaser_state, action, chaser_state_, reward)
            # Update current state
            chaser_state = chaser_state_
            if total_steps % 10 == 0:
                # Every 10 steps optimize model
                runner_loss = chaser.optimize_model()
            step += 1
            total_steps += 1
            if done:
                print('Episode: %d\tsteps: %d\tLoss: %f' %
                      (epi + 1, step + 1, runner_loss))
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
                    print('Episode: %d\tsteps: %d\tLoss: %f' %
                          (epi + 1, step + 1, runner_loss))
                    break
            # Update the target network, copying all weights and biases in DQN
            if total_steps % 500 == 0:
                # target net params replaced
                print('Target net params Replaced!')
                chaser.target_net.load_state_dict(
                    chaser.policy_net.state_dict())
                chaser.save('chaser-2000.pkl')


def greedy_chaser(env_name,
                  episode_count,
                  load_path,
                  display=True,
                  fps=10):
    """Show result of trained chaser to chase random agent runner.
    """
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
    # chaser_dqn(env, episode_count=40000, display=False, fps=100, need_reload=False,
    #            reload_path='./model/chaser-2000.pkl')
    # Observe result
    greedy_chaser(env, episode_count=1000, load_path='../../model/chaser-2000.pkl')
