import time

import gym
from grid_world.envs import AgentType

from agent.greedy_agent import GreedyAgent
from agent.greedy_probs_agent import GreedyProbsAgent


def runner_greedy(env_name,
                  episode_count,
                  policy_net,
                  chaser_restore_path,
                  display=True,
                  fps=10):
    """
    Both runner and chaser use their trained policy to run.
    To show how good the model are.
    """
    env = gym.make(env_name)
    chaser = GreedyAgent(default_reward=-1.0,
                         name='chaser',
                         color=(1.0, 0.0, 0.0),
                         env=env,
                         agent_type=AgentType.Chaser,
                         load_path=chaser_restore_path,
                         features_n=4)

    runner = GreedyProbsAgent(default_reward=0.5,
                              name='runner',
                              color=(0.0, 1.0, 0.0),
                              env=env,
                              agent_type=AgentType.Runner,
                              features_n=4,
                              policy_net=policy_net)
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
