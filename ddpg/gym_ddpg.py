from ddpg import *
import opensim as osim
from osim.http.client import Client
from osim.env import *

ENV_NAME = 'learning_to_run'
PATH = 'models/'
EPISODES = 100000
TEST = 1

def main():
    env = ei(True,seed=0,diff=0)
    env.reset()
    agent = DDPG(env)

    returns = []
    rewards = []

    rs = RunningStats()

    for episode in xrange(EPISODES):
        state = env.reset()
        reward_episode = []
        print "episode:",episode
        # Train
        demo = 50
        n_step = 3
        s,s1 = [],[]
        ea = engineered_action(np.random.rand())
        if np.random.rand() < 0.5:
            for i in range(demo):
                ob = env.step(ea)[0]
        ob = env.step(ea)[0]
        s = ob
        ob = env.step(ea)[0]
        s1 = ob
        s = process_state(s,s1,center=True) #s, which stands for state, is the new ob
        rs.normalize(s)
        for step in xrange(1000):
            ac = agent.action(s)
            print(ac)
            ac = np.clip(ac + agent.exploration_noise.noise(),0.05,0.95)
            temp = 0
            for i in range(n_step):
                ob, rew, new, _ = env.step(ac+agent.exploration_noise.noise()*0.2,0.05,0.95))
                rew = (rew/0.01 + int(new) * 0.1 + int((ob[2]/0.70)<1.0) * -1.)
                temp += rew
                if new: 
                    break
                s1 = ob
            rew = temp
            print(rew)
            s1 = process_state(s1,ob,center=True)
            rs.normalize(s1)
            agent.perceive(s,ac,rew,s1,new)
            s = s1
            s1 = ob
            reward_episode.append(rew)
            if new:
                break


        if episode % 5 == 0:
            print("episode reward = %.2f" % sum(reward_episode))
        # Testing:
        #if episode % 1 == 0:
        if episode % 100 == 0 and episode > 50:
            agent.save_model(PATH, episode)

            total_return = 0
            ave_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                reward_per_step = 0
                for i in range(demo):
                    ob = env.step(ea)[0]
                ob = env.step(ea)[0]
                s = ob
                ob = env.step(ea)[0]
                s1 = ob
                s = process_state(s,s1,center=True) #s, which stands for state, is the new ob
                rs.normalize(s)
                for j in xrange(1000):
                    ac = agent.action(s)
                    temp = 0
                    for i in range(n_step):
                        ob, rew, new, _ = env.step(ac)
                        rew = (rew/0.01 + int(new) * 0.1 + int((ob[2]/0.70)<1.0) * -1.)
                        temp += rew
                        if new: 
                            break
                        s1 = ob
                    rew = temp
                    s1 = process_state(s1,ob,center=True)
                    rs.normalize(s1)
                    s = s1
                    s1 = ob
                    total_return += rew
                    if new:
                        break
                    reward_per_step += (rew - reward_per_step)/(j+1)
                ave_reward += reward_per_step

            ave_return = total_return/TEST
            ave_reward = ave_reward/TEST
            returns.append(ave_return)
            rewards.append(ave_reward)

            print 'episode: ',episode,'Evaluation Average Return:',ave_return, '  Evaluation Average Reward: ', ave_reward

if __name__ == '__main__':
    main()
