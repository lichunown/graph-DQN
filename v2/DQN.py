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
input_size: [self.MAXVERTEX,self.MAXVERTEX]
    
output_size:[self.MAXVERTEX]

'''
class DQN(object):
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
#        model = Sequential()
#        model.add(Conv1D(self.train_batch//2 , 5,border_mode="valid",input_shape=self.input_size))
#        model.add(Flatten())
#        model.add(Dense(500, activation="relu"))
#        model.add(Dense(self.output_size, activation="linear"))
        
        gmodel = Sequential()
        gmodel.add(Conv1D(64 , 5,border_mode="valid",input_shape=self.input_size))
        gmodel.add(MaxPooling1D(pool_length=2, border_mode="valid"))
        gmodel.add(Dropout(0.3))
        gmodel.add(Conv1D(128 , 5,border_mode="valid"))
        gmodel.add(Dropout(0.3))
        gmodel.add(Flatten())
        vmodel = Sequential()
        vmodel.add(Dense(100, input_shape=(self.output_size,),activation="relu"))
        vmodel.add(Dense(100, activation="relu"))
        vmodel.add(Dense(100, activation="relu"))
        
        model = Sequential()
        model.add(Merge([gmodel, vmodel], mode="concat", concat_axis=-1))
        model.add(Dense(self.output_size, activation="linear"))
        
        model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=["accuracy"])
        return model
        
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


