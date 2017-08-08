# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


#from DQN_cart import DQN
import time
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Conv1D,LSTM,MaxPooling1D, Dropout, Flatten,Merge
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
import os
from keras.models import load_model
from keras import backend as K
from graph import GraphEnv

#agent = DQN(env)
#agent.load()
#agent.epsilon = 0
#
#for i_episode in range(3):
#    observation = env.reset()
#    while True:
#        env.render()
#        print(observation)
#        action = agent.action(observation)
#        observation, reward, done, info = env.step(action)
#        time.sleep(1/10)
#        if done:
#            print('DONE')
#            time.sleep(1)
#            break
'''
input_size: [self.MAXVERTEX,self.MAXVERTEX]
    
output_size:

'''
class DQN(object):
    def __init__(self,MAXVERTEX):
        self.input_size = (MAXVERTEX,MAXVERTEX)
        self.output_size = 2*MAXVERTEX
        self.memory = deque(maxlen=3000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.train_batch = 32
        self._inmodel = self._createModel()
        self._outmodel = self._createModel()
        
    @property
    def model(self):
        return self._model
    
    '''
    def createLSTMModel(self):# 定义训练模型
        qenc = Sequential()
        qenc.add(LSTM(self.QA_EMBED_SIZE, return_sequences=True,input_shape=(200,400)))
        qenc.add(Dropout(0.3))
        qenc.add(Conv1D(self.QA_EMBED_SIZE // 2, 5, border_mode="valid"))
        qenc.add(MaxPooling1D(pool_length=2, border_mode="valid"))
        qenc.add(Dropout(0.3))
        qenc.add(Flatten())
        aenc = Sequential()
        aenc.add(LSTM(self.QA_EMBED_SIZE, return_sequences=True,input_shape=(200,400)))
        aenc.add(Dropout(0.3))
        aenc.add(Conv1D(self.QA_EMBED_SIZE // 2, 3, border_mode="valid"))
        aenc.add(MaxPooling1D(pool_length=2, border_mode="valid"))
        aenc.add(Dropout(0.3))
        aenc.add(Flatten())
        _model = Sequential()
        _model.add(Merge([qenc, aenc], mode="concat", concat_axis=-1))
        _model.add(Dense(2, activation="softmax"))
        _model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=["accuracy"])
        return _model    
    '''
    def _createModel(self,dd=True): # TODO models
        model = Sequential()
        model.add(Conv1D(self.train_batch//2 , 5,border_mode="valid",input_shape=self.input_size))
        model.add(Flatten())
        model.add(Dense(200, activation="relu"))
        model.add(Dense(self.output_size//2, activation="linear"))
        model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=["accuracy"])
        return model
        
    def _findMaxIndex(self,r,info):
        return np.where(r*info!=0)[1][np.argmax((r*info)[np.where(r*info!=0)])]
    def train(self):
        if len(self.memory)>=self.train_batch:
            minibatch = random.sample(self.memory,self.train_batch) 
            state_batch = np.zeros([self.train_batch,self.input_size[0],self.input_size[1]])
            target_out_batch = np.zeros([self.train_batch,self.output_size//2]) 
            target_in_batch = np.zeros([self.train_batch,self.output_size//2]) 
            for i,(state, action, reward, next_state, done,info) in enumerate(minibatch):
                state_batch[i,:,:] = state
                action_out_num =  self._findMaxIndex(self.predict_action_out(state),info[0])#np.argmax(self.predict_action_out(state)[0]*info[0])
                action_in_num = self._findMaxIndex(self.predict_action_in(state),info[1])#np.argmax(self.predict_action_in(state)[0]*info[1])
                target_out_batch[i,:] = self.predict_action_out(state)[0]
                target_in_batch[i,:] = self.predict_action_in(state)[0]
                target_out_batch[i,action_out_num] = reward if done else reward+self.gamma*np.amax(self.predict_action_out(next_state)[0]*info[0])
                target_in_batch[i,action_in_num] = reward if done else reward+self.gamma*np.amax(self.predict_action_out(next_state)[0]*info[1])
                #print(target_out_batch)
            self._outmodel.fit(state_batch, target_out_batch, epochs=1, verbose=0)
            self._inmodel.fit(state_batch, target_in_batch, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            

    def predict_action_out(self,state):# 预测动作
        return self._outmodel.predict(state)
    def predict_action_in(self,state):# 预测动作
        return self._inmodel.predict(state)
    def predict_action(self,state):
        return np.concatenate([self.predict_action_out(state),self.predict_action_in(state)],1)
    
    def act(self,state):# 执行的动作，具有随机性
        if random.random() < self.epsilon:
            return np.random.random(self.output_size)
        else:
            #print(self.predict_action(state))
            return self.predict_action(state)[0]
        
    def remember(self,state,action,reward,next_state,done,info):
        self.memory.append((state,action,reward,next_state,done,info))
        #self._train()
    def save(self,name = 'models/test'):
        self.model.save(name)
        self.saveWeight(name)
    def load(self,name = 'models/test'):
        self._model= load_model(name)
    def saveWeight(self,name = 'models/test'):
        self.model.save_weights(name+'.weight')
    def loadWeight(self,name = 'models/test'):
        self.model.load_weights(name+'.weight')
        


MAXVERTEXNUM = 50

env = GraphEnv(50,5,MAXVERTEXNUM)
env.random(valuefun = np.random.random)

agent = DQN(MAXVERTEXNUM)

EPISODES = 1000
MAXTIMES = 500
epochs = 32

i = 0

for e in range(EPISODES):
    state = env.reset()
    state = state.reshape([1,MAXVERTEXNUM,MAXVERTEXNUM])
    lastreward = 0
    for times in range(MAXTIMES):
        action = agent.act(state)
        action.reshape([1,2*MAXVERTEXNUM])
        next_state,reward_pre,done,info = env.act(action)
        next_state = next_state.reshape([1,MAXVERTEXNUM,MAXVERTEXNUM])
        reward = 0.1 if reward_pre>=lastreward else -1# TODO change reward
        lastreward = reward_pre
        agent.remember(state,action,reward,next_state,done,info)
        state = next_state
        i += 1
        if done:
            print('{}|[times]:{}/{}    [i]:{}/{}    [reward_pre]:{}    [epsilon]:{:.2f}    [DONE]'.format(i,e+1,EPISODES,times,MAXTIMES,reward_pre,agent.epsilon))
            break
        if i % epochs==0:
            agent.train()
            print('{}|[times]:{}/{}    [i]:{}/{}    [reward_pre]:{}    [epsilon]:{:.2f}'.format(i,e+1,EPISODES,times,MAXTIMES,reward_pre,agent.epsilon))
        



