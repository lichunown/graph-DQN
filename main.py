# -*- coding: utf-8 -*-

from multiprocessing import cpu_count,Queue,Process,Lock
import numpy as np
import os
from queue import Empty
from env import GraphEnv
from DQN import DQN


if not os.path.exists('models/'):
    os.mkdir('models')

# args:
N = 20
MAXN = 50
M = 2
selectnum = 5
s2vlength = 100
GRAPHRANGE = 50

LOADWEIGHT = False


def modifyReward(lastr,reward): # use delta reward as the indicator of this step
    return reward - lastr


def envWorker(processi,inputqueue,outputqueue,s2vlock):
    print('[run] envWorker Process-%d'%processi)
    graphnum = 0
    while True:
        env = GraphEnv(n=N,m=M,s2vlength=s2vlength,maxSelectNum=selectnum,MAXN = MAXN)
        graphnum += 1
        s2vlock.acquire()  
        try:
            env.runs2v(maxsize = MAXN)
            print('[Process-{}] created struct'.format(processi))
        finally:
            s2vlock.release()
        for g in range(GRAPHRANGE):
            lastreward = 0
            state = env.reset()
            epsilon = None
            while True:
    #            action = agent.act(state)
#                print('[Process-{}] put act'.format(processi))
                outputqueue.put(('act',(state,epsilon)))
#                print('[Process-{}] get'.format(processi))
                action = inputqueue.get()
                state,action_onehot,reward,next_state,done = env.act(action)
                reward = modifyReward(lastreward,reward)
#                print('[Process-{}] put remember'.format(processi))
                outputqueue.put(('remember',(state, action_onehot, reward, next_state, done)))
    #            agent.remember(state, action_onehot, reward, next_state, done)
                state = next_state
                lastreward = reward
                if done:
                    break



                        
                        
                        
if __name__ == '__main__':
    results = []
    cmds = []
    envprocessnum = cpu_count() // 2
    s2vlock = Lock()
    for i in range(envprocessnum):
        results.append(Queue())
        cmds.append(Queue())
        p = Process(target=envWorker, args=(i,results[i],cmds[i],s2vlock))
        p.start()
        
    agent = DQN(MAXN = MAXN, s2vlength = s2vlength)
    remembertimes = 0
    traintimes = 0
    if LOADWEIGHT:
        try:
            agent.loadWeight()
        except Exception:
            pass   
        
    while True:
        for i in range(envprocessnum):
            try:
                cmd = cmds[i].get_nowait()
            except Empty:
                cmd = None
                pass
            if cmd:
#                print('[main] #for {}# get cmd: {}'.format(i,cmd[0]))
                if cmd[0]=='act':
#                    print('[main] #for {}# put act'.format(i))
                    results[i].put(agent.act(cmd[1][0],cmd[1][1]))
                elif cmd[0]=='remember':
                    agent.remember(cmd[1][0],cmd[1][1],cmd[1][2],cmd[1][3],cmd[1][4])
                    remembertimes += 1
                    if remembertimes >= 16:
                        remembertimes = 0
                        agent.train()
                        traintimes += 1
                        if traintimes >= 10:
                            agent.saveWeight()
                        print('[main] agent Train epslion: {}'.format(agent.epsilon))
            
    
    
    
    
    
    
    

