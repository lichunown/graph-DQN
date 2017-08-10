# -*- coding: utf-8 -*-

import numpy as np
from graph import GraphEnv
from DQN import DQN


MAXVERTEXNUM = 50

env = GraphEnv(50,5,MAXVERTEXNUM)
env.load('test')
#env.random(valuefun = np.random.random)

agent = DQN(MAXVERTEXNUM)

EPISODES = 10000
MAXTIMES = 500
epochs = 32

i = 0
maxValue = 0
maxValue_Vertex = None

for e in range(EPISODES):
    state = env.reset()
    state = state.reshape([1,MAXVERTEXNUM,MAXVERTEXNUM])
    lastreward = 0  
    for times in range(MAXTIMES):
        action = agent.act(state)
        action.reshape([1,2*MAXVERTEXNUM])
        next_state,reward_pre,done,info = env.act(action)
        next_state = next_state.reshape([1,MAXVERTEXNUM,MAXVERTEXNUM])
        reward = 10 if reward_pre>=lastreward else -1 # TODO change reward
        if reward_pre > maxValue:
            maxValue = reward_pre
            maxValue_Vertex = env.selectVertex
        lastreward = reward_pre
        agent.remember(state,action,reward,next_state,done,info)
        state = next_state
        i += 1
        if done:
            print('{}|[EPISODES]:{}/{}    [times]:{}/{}    [max_reward]:{}    [reward_pre]:{}    [epsilon]:{:.2f}'.format(i,e+1,EPISODES,times,MAXTIMES,maxValue,reward_pre,agent.epsilon))
            break
        if i % epochs==0:
            agent.train()
            print('{}|[EPISODES]:{}/{}    [times]:{}/{}    [max_reward]:{}    [reward_pre]:{}    [epsilon]:{:.2f}'.format(i,e+1,EPISODES,times,MAXTIMES,maxValue,reward_pre,agent.epsilon))
            agent._inmodel.save_weights('inmodel.weight')
            agent._outmodel.save_weights('outmodel.weight')