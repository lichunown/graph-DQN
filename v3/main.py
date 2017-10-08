# -*- coding: utf-8 -*-

import numpy as np

from env import GraphEnv
from DQN import DQN



MAXE = 20
GRAN = 50
agent = DQN(MAXN = 50)

for gtimes in range(GRAN):
    env = GraphEnv(n=20,m=2,s2vlength=100,maxSelectNum=5,MAXN = 50)
    for e in range(MAXE):
        state = env.reset()
        times = 0
        lastreward = 0
        while True:
            times += 1 
            action = agent.act(state)
            state,action_onehot,reward,next_state,done = env.act(action)
            agent.remember(state,action_onehot,reward - lastreward,next_state,done)
            state = next_state
            lastreward = reward
            agent.train()
#            print("g:{}/{} e:{}/{}  times:{}  reward:{:.2}  epsilon:{:.2}  predict:{:.2}".format(gtimes,GRAN,e,MAXE,
#                                                  times,reward,agent.epsilon,
#                                                  agent.predict_action_value(state)))
            if done:
                print("g:{}/{} e:{}/{}  times:{}  reward:{}  epsilon:{:.2}  predict:{:.2}".format(gtimes,GRAN,e,MAXE,
                                                  times,reward,agent.epsilon,
                                                  agent.predict_action_value(state)))
                break
    
