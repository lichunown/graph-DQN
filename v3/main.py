# -*- coding: utf-8 -*-

import numpy as np

from env import GraphEnv
from DQN import DQN



MAXE = 50



for e in range(MAXE):
    env = GraphEnv(n=10,m=2,s2vlength=100,maxSelectNum=3,MAXN = 10)
    state = env.reset()
    times = 0
    lastreward = 0
    while True:
        times += 1 
        action = agent.act(state)
        state,action_onehot,reward,next_state,done = env.act(action)
        agent.remember(state,action_onehot,reward/10,next_state,done)
        state = next_state
        lastreward = reward
        agent.train()
        print("e:{}/{}  times:{}  reward:{:.2}  epsilon:{:.2}  predict:{:.2}".format(e,MAXE,
                                              times,reward,agent.epsilon,
                                              agent.predict_action_value(state)))
        if done:
            print('DONE e:{}/{} reward:{:.2}'.format(e,MAXE,reward))
            break

