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
        self._model = self._createModel()
    
    def epsilondecay(self):
        self.epsilon -= (1-self.epsilon_min)/(self.MAXN*self.MAXN/self.train_batch)
        
    @property
    def model(self):
        return self._model
    
    def createLSTMModel(self):# 定义训练模型
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
    
    def train(self):
        if len(self.memory)>=self.train_batch:
            minibatch = random.sample(self.memory,self.train_batch) 
            state_batch_g = np.zeros([self.train_batch,self.input_size[0],self.input_size[1]])
            state_batch_v = np.zeros([self.train_batch,self.output_size]) 
            target_batch = np.zeros([self.train_batch,self.output_size]) 
            #target_in_batch = np.zeros([self.train_batch,self.output_size]) 
            for i,((state_g,state_v), action, reward, (next_state_g,next_state_v), done,info) in enumerate(minibatch):
                state_batch_g[i,:,:] = state_g
                state_batch_v[i,:] = state_v
                action_num =  self._findMaxIndex(self.predict_action([state_g,state_v])[0],info)#np.argmax(self.predict_action_out(state)[0]*info[0])
                target_batch[i,:] = self.predict_action([state_g,state_v])[0]
                target_batch[i,action_num] = reward if done else reward+self.gamma*np.amax(self.predict_action([next_state_g,next_state_v])[0]*info)

            self._model.fit([state_batch_g,state_batch_v], target_batch, epochs=1, verbose=0)
            #self._inmodel.fit(state_batch, target_in_batch, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            

#    def predict_action_out(self,state):# 预测动作
#        return self._outmodel.predict(state)
#    def predict_action_in(self,state):# 预测动作
#        return self._inmodel.predict(state)
    def predict_action(self,state):
        return self.model.predict(state)
    
    def act(self,state):# 执行的动作，具有随机性
        if random.random() < self.epsilon:
            return np.random.random(self.MAXN)
        else:
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



class DQN_test(object):
    def __init__(self,MAXVERTEX):
        self.input_size = (MAXVERTEX,MAXVERTEX)
        self.output_size = MAXVERTEX
        self.memory = deque(maxlen=3000)
        self.gamma = 1    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.train_batch = 32
        self._model = self._createModel()
        
    @property
    def model(self):
        return self._model
    
    def _createModel(self,dd=True): # test, input shape too small   
        gmodel = Sequential()
        gmodel.add(Flatten( input_shape=self.input_size))
        gmodel.add(Dense(200, activation="relu"))
        gmodel.add(Dense(200, activation="relu"))
        gmodel.add(Dense(200, activation="relu"))
        vmodel = Sequential()
        vmodel.add(Dense(100, input_shape=(self.output_size,),activation="relu"))
        vmodel.add(Dense(100, activation="relu"))
        vmodel.add(Dense(100, activation="relu"))
        
        model = Sequential()
        model.add(Merge([gmodel, vmodel], mode="concat", concat_axis=-1))
        model.add(Dense(self.output_size, activation="linear"))
        
        model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=["accuracy"])
        return model
        
    def predict_action_num(self,r,info):
        n = np.where(r*info!=0)[0][np.argmax((r*info)[np.where(r*info!=0)])]
        return (n,r[n])
    
    def train(self):
        if len(self.memory)>=self.train_batch:
            minibatch = random.sample(self.memory,self.train_batch) 
            state_batch_g = np.zeros([self.train_batch,self.input_size[0],self.input_size[1]])
            state_batch_v = np.zeros([self.train_batch,self.output_size]) 
            target_batch = np.zeros([self.train_batch,self.output_size]) 
            #target_in_batch = np.zeros([self.train_batch,self.output_size]) 
            for i,((state_g,state_v), action, reward, (next_state_g,next_state_v), done,info) in enumerate(minibatch):
                state_batch_g[i,:,:] = state_g
                state_batch_v[i,:] = state_v
                action_num,v = self.predict_action_num(self.predict_action([state_g,state_v])[0],info)#np.argmax(self.predict_action_out(state)[0]*info[0])
                #print(v)
                target_batch[i,:] = self.predict_action([state_g,state_v])[0]
                target_batch[i,action_num] = reward if done else reward+self.gamma*np.amax(self.predict_action([next_state_g,next_state_v])[0]*info)

            self._model.fit([state_batch_g,state_batch_v], target_batch, epochs=1, verbose=0)
            #print(self._model.predict([state_batch_g,state_batch_v]))
            #self._inmodel.fit(state_batch, target_in_batch, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            

#    def predict_action_out(self,state):# 预测动作
#        return self._outmodel.predict(state)
#    def predict_action_in(self,state):# 预测动作
#        return self._inmodel.predict(state)
    def predict_action(self,state):
        return self.model.predict(state)
    
        
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
#
#MAXVERTEXNUM = 50
#agent = DQN(MAXVERTEXNUM)
#for i in range(32):
#    agent.remember([np.random.random([1,MAXVERTEXNUM,MAXVERTEXNUM]),np.random.random([1,MAXVERTEXNUM])],
#                   np.random.random([MAXVERTEXNUM]),
#                   np.random.random(),
#                   [np.random.random([1,MAXVERTEXNUM,MAXVERTEXNUM]),np.random.random([1,MAXVERTEXNUM])],
#                   random.sample([True,False],1),
#                   np.random.random([MAXVERTEXNUM]),
#                   )
#
#agent.train()  
#agent.predict_action(np.random.random([1,MAXVERTEXNUM,MAXVERTEXNUM]))


