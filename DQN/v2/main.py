# -*- coding: utf-8 -*-

import numpy as np
import random
from graph import GraphEnv
from DQN import DQN,DQN_test

################
import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='train.log',
                filemode='w')


#################
MAXVERTEXNUM = 10
VERTEXNUM = 10
SELECTNUM = 3


env = GraphEnv(VERTEXNUM,SELECTNUM,MAXVERTEXNUM)

logging.info('test')

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
def randomTest(env,loops=500):
    maxValue = 0
    maxValue_vertex = np.zeros(VERTEXNUM)
    maxValue_num = 0
    for i in range(loops):
        env.reset()
        onelist = random.sample(range(VERTEXNUM),SELECTNUM)
        r = np.zeros(VERTEXNUM)
        r[onelist] = 1
        env.selectVertex = r
        if env.reward_pre > maxValue:
            if not np.all(maxValue_vertex == env.selectVertex):
                maxValue = env.reward_pre
                maxValue_vertex = env.selectVertex
                maxValue_num = i+1
    print('[randomTest]: running {} times. Found [maxValue]:{}   [maxValue_num]:{}'.format(loops,
                                                      maxValue,maxValue_num))
    print('maxValue_vertex: '+str(maxValue_vertex))
    logging.info('[randomTest]: running {} times. Found [maxValue]:{}   [maxValue_num]:{}'.format(loops,
                                                      maxValue,maxValue_num))
    logging.info('maxValue_vertex: '+str(maxValue_vertex))
    return (maxValue,maxValue_vertex,maxValue_num)
            
    







logging.info('\n\n[Start]')

agent = DQN_test(MAXVERTEXNUM)
try:
    agent.loadWeight('outmodel')
    logging.info('load weight')
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


IFEXPECT = True
RANDOMTEST = True
for g in range(GRAPHTYPE):
    env.random(valuefun = np.random.random)
    logging.info('random Graph')
    if IFEXPECT:
        expect,expectmaxVertex = env.initMaxValue() 
        logging.info('[expect]:%f'%expect)
        logging.info('[expectmaxVertex]:'+str(expectmaxVertex))
    if RANDOMTEST:
        randomTest(env)
    maxValue = 0
    maxValue_Vertex = np.zeros(VERTEXNUM)
    findmax_times = None        
    for e in range(EPISODES):
        state = env.reset()
        lastreward = 0  
        for times in range(MAXTIMES):
            action = agent.act(state)
            action.reshape([1,MAXVERTEXNUM])          
            next_state,reward_pre,done,info = env.act(action)
            reward = reward_pre
            if reward_pre > maxValue:
                if not np.all(maxValue_Vertex==env.selectVertex):
                    maxValue = reward_pre
                    maxValue_Vertex = env.selectVertex
                    findmax_times = e
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
                    print('{}|[G]:{}/{}   [EPISODES]:{}/{}   [times]:{}/{}   [expect]:{}   [max_reward]:{}   [reward_pre]:{}   [epsilon]:{:.2f}'\
                      .format(i,g+1,GRAPHTYPE,e+1,EPISODES,times+1,MAXTIMES,expect,maxValue,reward_pre,                  \
                       agent.epsilon))       
                else:
                    print('{}|[G]:{}/{}   [EPISODES]:{}/{}   [times]:{}/{}   [max_reward]:{}   [reward_pre]:{}   [epsilon]:{:.2f}'\
                      .format(i,g+1,GRAPHTYPE,e+1,EPISODES,times+1,MAXTIMES,maxValue,reward_pre,                  \
                       agent.epsilon))                       
                break
    logging.info('DQN [maxValue]:{}  [findmax_times]:{}'.format(maxValue,findmax_times))
    logging.info('[maxValue_Vertex]:'+str(maxValue_Vertex))

                
