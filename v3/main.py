# -*- coding: utf-8 -*-

import numpy as np

from env import GraphEnv
from DQN import DQN



MAXE = 50

agent = DQN(MAXVERTEX = 50)

for e in range(MAXE):
    env = GraphEnv(n=10,m=2,s2vlength=100,maxSelectNum=3,MAXN = 50)
    state = env.reset()
    times = 0
    while True:
        times += 1 
        action = agent.act(state)
        state,action_onehot,reward,next_state,done = env.act(action)
        agent.remember(state,action_onehot,reward,next_state,done)
        state = next_state
        agent.train()
        print("e:{}/{}  times:{}  reward:{}".format(e,MAXE,times,reward))
        if done:
            break

