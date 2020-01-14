import time

import gym
from puck_world.envs import AgentType

from puck_world_wheel.dqn_agent import EGreedyAgent
from puck_world_wheel.greedy_agent import GreedyAgent
from puck_world_wheel.random_agent import RandomAgent


def chaser_dqn(env_name,
               episode_count=1000,
               close=True,
               need_reload=False):
    """Use DQN to train Chaser to chase a random runner agent.
    """
    env = gym.make(env_name)
    # Epsilon Greedy agent with network to make policy
    chaser = EGreedyAgent(50,
                          50,
                          30,
                          color=(1, 0, 0),
                          agent_type=AgentType.Chaser,
                          features_n=5,
                          actions_n=4,
                          discounted_value=0.99,
                          need_restore=need_reload)

    # Randomly make choice
    runner = RandomAgent(300,
                         300,
                         30,
                         color=(0, 1, 0),
                         agent_type=AgentType.Runner,
                         env=env)

    env.add_agent(chaser)
    env.add_agent(runner)

    reward = 0
    done = False

    # Total steps contains every episode's steps
    total_steps = 0

    for epi in range(episode_count):
        chaser_state = env.reset()
        chaser_state.append(chaser.direction)

        # current episode's steps
        step = 0
        while True:
            env.render(close=close)
            left_v, right_v = chaser.act(chaser_state)
            chaser.set_left_v(left_v)
            chaser.set_right_v(right_v)
            action = (left_v, right_v)
            chaser_state_, reward, done, _ = env.step(chaser.type, action[0])
            chaser_state_.append(chaser.direction)

            # Put this transition into memory
            chaser.memory.push(chaser_state, action, chaser_state_, reward)
            # Update current state
            chaser_state = chaser_state_
            if total_steps % 100 == 0:
                # Every 10 steps optimize model
                chaser_loss = chaser.optimize_model()
            step += 1
            total_steps += 1
            if done:
                print('Chaser\tEpisode: %d\tsteps: %d\tLoss: %f' %
                      (epi + 1, step + 1, chaser_loss))
                break
            else:
                runner_action = runner.act()
                runner_state_, reward, done, _ = env.step(runner.type, runner_action)
                env.render(close=close)
                step += 1
                if done:
                    # 添加 正值 reward 到 集合中
                    # a = get_reverse_action(runner_action)
                    # s = chaser_state
                    # r = 10
                    # s_ = [chaser_x, chaser_y, runner_x, runner_y]
                    # chaser.memory.push(s, a, s_, r)
                    print('Episode: %d\tsteps: %d\tLoss: %f' %
                          (epi + 1, step + 1, chaser_loss))
                    break
            # Update the target network, copying all weights and biases in DQN
            if total_steps % 10000 == 0:
                # target net params replaced
                print('Target net params Replaced!')
                chaser.target_net.load_state_dict(chaser.policy_net.state_dict())
                chaser.save()


def greedy_chaser(env_name,
                  episode_count,
                  close=True):
    """Show result of trained chaser to chase random agent runner.
    """
    env = gym.make(env_name)
    chaser = GreedyAgent(50,
                         50,
                         30,
                         color=(1, 0, 0),
                         agent_type=AgentType.Chaser,
                         features_n=5,
                         actions_n=4)

    runner = RandomAgent(300,
                         300,
                         30,
                         color=(0, 1, 0),
                         agent_type=AgentType.Runner,
                         env=env)

    env.add_agent(chaser)
    env.add_agent(runner)

    for epi in range(episode_count):
        chaser_state = env.reset()
        chaser_state.append(chaser.direction)

        step = 0
        while True:
            env.render(close=close)
            left_v, right_v = chaser.act(chaser_state)
            chaser.set_right_v(right_v)
            chaser.set_left_v(left_v)
            action = (left_v, right_v)
            chaser_state_, reward, done, _ = env.step(chaser.type, action[0])
            chaser_state = chaser_state_
            chaser_state.append(chaser.direction)
            step += 1
            if done:
                print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                break
            else:
                runner_action = runner.act()
                runner_state_, _, done, _= env.step(runner.type, runner_action)
                env.render(close=close)
                runner_state = runner_state_
                step += 1
                if done:
                    print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                    break


if __name__ == '__main__':
    env = 'puck-world-wheel-v0'
    chaser_dqn(env, episode_count=20000, close=True, need_reload=True)
    # Observe result
    # greedy_chaser(env, episode_count=1000, close=False)
