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
from .graph import GraphEnv

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
        self._model = self._createModel()
        
    @property
    def model(self):
        return self._model
    
    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
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
        gmodel = Sequential()
        gmodel.add(Embedding(input_dim = self.input_size, output_dim= 20))
        gmodel.add(Dense(self.output_size, activation="softmax"))
        gmodel.compile(optimizer="adam", loss='categorical_crossentropy',metrics=["accuracy"])
        return gmodel
        
    def train(self):
        if len(self.memory)>=self.train_batch:
            minibatch = random.sample(self.memory,self.train_batch) 
            state_batch = np.zeros([self.train_batch,self.state_size])
            target_batch = np.zeros([self.train_batch,self.action_size]) 
            for i,(state, action, reward, next_state, done) in enumerate(minibatch):
                state_batch[i,:] = state
                target_batch[i,:] = self.predict_action(state)
                target_batch[i,action] = reward if done else reward+self.gamma*np.amax(self.predict_action(next_state)[0])
            self.model.fit(state_batch, target_batch, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
    def predict_action(self,state):# 预测动作
        return self.model.predict(state)
    def act(self,state):# 执行的动作，具有随机性
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            #print(self.predict_action(state))
            return np.argmax(self.predict_action(state)[0])
        
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
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
        


MAXVERTEXNUM = 100

env = GraphEnv(20,5,MAXVERTEXNUM)
env.random(valuefun = np.random.random)

agent = DQN(MAXVERTEXNUM)

EPISODES = 1000
MAXTIMES = 5000
epochs = 32
for e in range(EPISODES):
    state = env.reset()
    for times in range(MAXTIMES):
        action = agent.act(state)
        next_state,reward_pre,done = env.act(action)
        reward = reward_pre# TODO change reward
        agent.remember(state,action,reward,next_state,done)
        state = next_state
        if done:
            print('[times]:{}/{}\t\t[i]:{}\t[reward]:{}\t[epsilon]:{}'.format(e,EPISODES,times,reward,agent.epsilon))
            break
        if times % epochs==0:
            agent.train()
        



