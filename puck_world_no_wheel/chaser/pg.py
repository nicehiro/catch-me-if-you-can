import gym
from puck_world_no_wheel.envs import puck_world, AgentType, Agent
from pg_agent import ActorCriticNet, ActorCriticAgent
from random_agent import RandomAgent


def train_chaser(env_name,
                 chaser_need_restore,
                 episode_count,
                 close=False):
    """
    Use Actor-Critic TD(0) method to train runner.

    Cannot use batch method to optimize model.
    """
    env = gym.make(env_name)
    features_n = 4

    chaser = ActorCriticAgent(300,
                              300,
                              30,
                              color=(1.0, 0.0, 0.0),
                              agent_type=AgentType.Chaser,
                              features_n=features_n,
                              discounted_value=0.9,
                              learning_rate=1e-6,
                              need_restore=chaser_need_restore)

    runner = RandomAgent(500,
                         500,
                         30,
                         color=(0.0, 1.0, 0.0),
                         agent_type=AgentType.Runner,
                         env=env)

    env.add_agent(chaser)
    env.add_agent(runner)

    total_steps = 0

    losses = []
    for epi in range(episode_count):
        chaser_state = env.reset()

        step = 0
        while True:
            env.render(close=close)
            action = chaser.act(chaser_state)
            chaser_state_, reward, done, _ = env.step(chaser.type, action)
            chaser.memory.append((chaser_state, action, chaser_state_, reward, done))
            chaser_state = chaser_state_
            step += 1
            total_steps += 1
            if step >= 500:
                # When episode's steps above 500, break up this episode
                chaser.memory.clear()
                break
            if done:
                loss = chaser.optimize_model()
                print('Episode: %d\tsteps: %d\tloss: %f' % (epi + 1, step + 1, loss))
                losses.append(loss)
                break
            else:
                runner_action = runner.act(chaser_state)
                _, _, done, _ = env.step(runner.type, runner_action)
                env.render(close=close)
                step += 1
                total_steps += 1
                if done:
                    chaser.memory.clear()
                    print('Episode: %d\tsteps: %d' % (epi + 1, step + 1))
                    break
        # 每个回合结束保存模型
        if epi % 100 == 0:
            chaser.save()


if __name__ == '__main__':
    env = 'puck-world-v0'
    train_chaser(env,
                 chaser_need_restore=False,
                 episode_count=20000,
                 close=False)
    # ac_net = ActorCriticNet(4, 4)
    # ac_net.load_state_dict(torch.load('../../model/ac.pkl'))
    # ac_net = ac_net.pi
    #
    # runner_greedy(env,
    #               episode_count=100,
    #               chaser_restore_path='../../model/chaser-2000.pkl',
    #               display=True,
    #               fps=10,
    #               policy_net=ac_net)
