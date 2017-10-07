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


'''
self.n = N
input_size: ([self.n],[self.n,self.s2vlength])
    
output_size:[self.MAXVERTEX]

'''
class DQN(object):
    def __init__(self,MAXVERTEX):
        self.MAXN = MAXVERTEX
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.train_batch = 32
        self._model = self.createLSTMModel()
    
    def epsilondecay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (1-self.epsilon_min)/(self.MAXN*self.MAXN/self.train_batch)
        
    @property
    def model(self):
        return self._model
    
    def createLSTMModel(self):# TODO 定义训练模型
        gm = Sequential()
        gm.add(LSTM(32, return_sequences=True,input_shape=(200,400)))
        gm.add(Dropout(0.3))
        gm.add(Conv1D(64, 5, border_mode="valid"))
        gm.add(MaxPooling1D(pool_length=2, border_mode="valid"))
        gm.add(Dropout(0.3))
        gm.add(Conv1D(128, 5, border_mode="valid"))
        gm.add(MaxPooling1D(pool_length=2, border_mode="valid"))
        gm.add(Dropout(0.3))
        gm.add(Flatten())
        sm = Sequential()
        sm.add(Dense(128, input_shape=(self.MAXN,),activation="relu"))
        sm.add(Dense(256,activation="relu"))

        _model = Sequential()
        _model.add(Merge([sm, gm], mode="concat", concat_axis=-1))
        _model.add(Dense(256,activation="relu"))
        _model.add(Dense(256,activation="linear"))
        _model.add(Dense(512,activation="linear"))
        _model.add(Dense(self.MAXN, activation="linear"))
        _model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=["accuracy"])
        return _model    
 
    
    
    
    def _findMaxIndex(self,r,info):
        return np.where(r*info!=0)[0][np.argmax((r*info)[np.where(r*info!=0)])]
    
    def train(self):# TODO
        if len(self.memory)>=self.train_batch:
            minibatch = random.sample(self.memory,self.train_batch) 
            for state,action,reward,next_state,done in minibatch:
                target = reward if done else (reward + self.gamma * self.predict_action(next_state)[1])             
                target_f = self.model.predict(state)
                target_f[0][action] = target                
                self.model.fit(state, target_f, epochs=1, verbose=0)
            self.epsilondecay()


    def predict_action_onehot(self,state):
        return self.model.predict(state)[0]

    def predict_action(self,state):
        act_onehot = self.predict_action_onehot(state)
        enableselect = np.zeros(len(state[0]))
        enableselect[np.where(state[0]==0)[0]] = 1
        action = np.argmax(act_onehot*enableselect)
        if act_onehot[action]==0:
            raise('Best reward of selection is 0. ({})'.format(act_onehot))
        return action,act_onehot[action]
    
    def act(self,state):# 执行的动作，具有随机性
        if random.random() < self.epsilon:
            return random.sample(list(np.where(state[0]==0)[0]),1)[0]
        else:
            return self.predict_action(state)
        
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



