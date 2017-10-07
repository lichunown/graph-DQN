# -*- coding: utf-8 -*-

import numpy as np

from env import GraphEnv
from DQN import DQN



MAXE = 3

agent = DQN(MAXVERTEX = 11)

for e in range(MAXE):
    env = GraphEnv(n=10,m=2,s2vlength=100,maxSelectNum=3,MAXN = 11)
    state = env.reset()
    times = 0
    while True:
        times += 1 
        action = agent.act(state)
        state,action_onehot,reward,next_state,done = env.act(action)
        agent.remember(state,action_onehot,reward,next_state,done)
        agent.train()
        if done:
            print("e:{}/{}  times:{}  reward:{}".format(e,MAXE,times,reward))
            break
#    state_size = env.observation_space.shape[0]
#    action_size = env.action_space.n
#    agent = DQNAgent(state_size, action_size)
#    agent.load("./models/cartpole-dqn-keras.h5")
#    for e in range(3):
#        state = env.reset()
#        state = np.reshape(state, [1, state_size])
#        times = 0
#        while True:
#            # env.render()
#            action = agent.act(state)
#            next_state, reward, done, _ = env.step(action)
#            next_state = np.reshape(next_state, [1, state_size])
#            state = next_state
#            times += 1
#            time.sleep(1/60)
#            if done:
#                print("[Fail] health:{} times:{}".format(e, times))
#                break

    