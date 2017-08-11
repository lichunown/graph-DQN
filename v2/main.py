# -*- coding: utf-8 -*-

import numpy as np
from graph import GraphEnv
from DQN import DQN


MAXVERTEXNUM = 10

VERTEXNUM = 10
SELECTNUM = 3


env = GraphEnv(VERTEXNUM,SELECTNUM,MAXVERTEXNUM)

env.load('test')
#env.random(valuefun = lambda x:0.2)
#env.random(valuefun = np.random.random)

agent = DQN(MAXVERTEXNUM)

EPISODES = 100000
MAXTIMES = SELECTNUM
savetimes = 32
traintimes = 1

i = 0
maxValue = 0
maxValue_Vertex = None

for e in range(EPISODES):
    state = env.reset()
    lastreward = 0  
    for times in range(MAXTIMES):
        action = agent.act(state)
        action.reshape([1,MAXVERTEXNUM])
        next_state,reward_pre,done,info = env.act(action)
        reward = reward_pre
        #reward = 10 if reward_pre>=lastreward else -1 # TODO change reward
        if reward_pre > maxValue:
            maxValue = reward_pre
            maxValue_Vertex = env.selectVertex
        lastreward = reward_pre
        agent.remember(state,action,reward,next_state,done,info)
        state = next_state
        i += 1
        if i % traintimes==0:
            agent.train()
        if i % savetimes==0:          
            agent.model.save_weights('outmodel.weight')
        if done:
            print('{}|[EPISODES]:{}/{}    [times]:{}/{}    [max_reward]:{}    [reward_pre]:{}    [tempTrueReward]:{:.2f}    [epsilon]:{:.2f}'\
                  .format(i,e+1,EPISODES,times+1,MAXTIMES,maxValue,reward_pre,                  \
                   action[agent._findMaxIndex(action,info)],agent.epsilon))              
            break