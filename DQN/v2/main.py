# -*- coding: utf-8 -*-

import numpy as np
from graph import GraphEnv
from DQN import DQN,DQN_test


MAXVERTEXNUM = 10
VERTEXNUM = 10
SELECTNUM = 3


env = GraphEnv(VERTEXNUM,SELECTNUM,MAXVERTEXNUM)



#agent = DQN_test(MAXVERTEXNUM)
#
#MAXTIMES = SELECTNUM
#EPISODES = int(np.ceil(32/MAXTIMES))
#env.random(valuefun = np.random.random)
#
#for e in range(EPISODES):
#    state = env.reset()
#    lastreward = 0  
#    for times in range(MAXTIMES):
#        action = agent.act(state)
#        action.reshape([1,MAXVERTEXNUM])          
#        next_state,reward_pre,done,info = env.act(action)
#        reward = reward_pre
#        lastreward = reward_pre
#        agent.remember(state,action,reward,next_state,done,info)
#        state = next_state
#agent.train()










agent = DQN_test(MAXVERTEXNUM)
try:
    agent.loadWeight('outmodel')
    print('[Message] load weight')
except Exception:
    print('[Message] NOT load weight')
    pass

GRAPHTYPE = 2000
EPISODES = 500
MAXTIMES = SELECTNUM
savetimes = 32
traintimes = 1

i = 0
maxValue = 0
maxValue_Vertex = None

IFEXPECT = False

for g in range(GRAPHTYPE):
    env.random(valuefun = np.random.random)
    if IFEXPECT:
        expect,expectmaxVertex = env.initMaxValue()   
    for e in range(EPISODES):
        state = env.reset()
        lastreward = 0  
        for times in range(MAXTIMES):
            action = agent.act(state)
            action.reshape([1,MAXVERTEXNUM])          
            next_state,reward_pre,done,info = env.act(action)
            reward = reward_pre
            if reward_pre > maxValue:
                maxValue = reward_pre
                maxValue_Vertex = env.selectVertex
            lastreward = reward_pre
            agent.remember(state,action,reward,next_state,done,info)
            state = next_state
            i += 1
            agent.train()
            if i % traintimes==0:
                agent.train()
            if i % savetimes==0:          
                agent.model.save_weights('outmodel.weight')
            if done:
                if IFEXPECT:
                    print('{}|[G]:{}/{}   [EPISODES]:{}/{}   [times]:{}/{}   [expect]:{}   [max_reward]:{}   [reward_pre]:{}   [temp]:{:.2f}   [epsilon]:{:.2f}'\
                      .format(i,g+1,GRAPHTYPE,e+1,EPISODES,times+1,MAXTIMES,expect,maxValue,reward_pre,                  \
                       action[agent._findMaxIndex(action,info)],agent.epsilon))       
                else:
                    print('{}|[G]:{}/{}   [EPISODES]:{}/{}   [times]:{}/{}   [max_reward]:{}   [reward_pre]:{}   [epsilon]:{:.2f}'\
                      .format(i,g+1,GRAPHTYPE,e+1,EPISODES,times+1,MAXTIMES,maxValue,reward_pre,                  \
                       agent.epsilon))                       
                break
                
