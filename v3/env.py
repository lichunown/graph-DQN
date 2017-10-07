# -*- coding: utf-8 -*-
import numpy as np
import random
from itertools import combinations
#from scipy.special import comb
import networkx as nx

class Graph(object):
    def __init__(self,n = 10,m = 2,s2vlength = None):
        self._graph = nx.barabasi_albert_graph(n,m)
        self.n = n
        self.m = m
        self.s2vlength = s2vlength
    @property
    def graph(self):
        return self._graph
    
    def draw(self):
        nx.draw(self.graph)
        
    def out(self,maxsize=None):
#        return self.s2v()
        return self.out_array(maxsize)
    
    def out_array(self,maxsize = None):
        if not maxsize:
            return nx.to_numpy_array(self.graph)
        else:
            assert maxsize >= self.n
            result = -1*np.ones([maxsize,maxsize])
            temp = nx.to_numpy_array(self.graph)
            result[[[i] for i in range(self.n)],[list(range(self.n)) for i in range(self.n)]] = temp
            return result
        
    def isConnected(self,n,v):
        return self._graph.has_edge(n,v)
    
    def save(self,filename):
        pass
        
    def load(self,filename):
        pass
    
    def s2v(self):
        pass
    
    @property
    def edges(self):
        return self.graph.edges
    @property
    def nodes(self):
        return self.graph.nodes
    
    def __str__(self):
        return str(self.graph.__str__())
    def __repr__(self):
        return str(self.graph.__repr__())
    def __len__(self):
        return self.graph.__len__()



class GraphEnv(Graph):
    def __init__(self,n = 10,m = 2,s2vlength = 100,maxSelectNum = 5,MAXN = None):
        super(GraphEnv,self).__init__(n,m,s2vlength)      
        self.selectset = set()
        self.notselectset = set(self.graph.nodes)
        self.MAXSELECTNUM = maxSelectNum
        if MAXN:
            assert MAXN >= n
        self.MAXN = MAXN if MAXN else None     
        self.REWARDRUNTIMES = 233
        self.casRate = 0.5
        assert maxSelectNum <= self.n
    def selectVertex(self,maxsize = None):# 选中的顶点
        if not maxsize:
            r = np.zeros(self.n)
            r[list(self.selectset)] = 1
        else:
            r = -1*np.ones(maxsize)
            r[list(self.selectset)] = 1
            r[list(self.notselectset)] = 0
        return r
    
    def select(self,num:int):
        assert num < self.n
        assert num in self.notselectset
        assert len(self.selectset) <= self.MAXSELECTNUM
        self.notselectset.remove(num)
        self.selectset.add(num)
    
    def select_list(self,nums):
        for num in nums:
            self.select(num)
            
    def reset(self):# 重置，为0
        self.selectset = set()
        self.notselectset = set(self.graph.nodes)
        return self.state(self.MAXN)
    
    def act(self,action:int,maxsize=None):
        state = self.state(self.MAXN)
        self.select(action)
        action_onehot = -1 * np.ones(self.MAXN)
        action_onehot[0:self.n] = 0
        action_onehot[action] = 1
        return state,action_onehot,self.reward,self.state(self.MAXN),self.done
    

    def state(self,maxsize = None):
        return self.selectVertex(maxsize),self.out(maxsize)
    
    @property
    def reward(self):
        return self.reward_pre2
    
    @property
    def done(self) -> bool:
        return len(self.selectset) == self.MAXSELECTNUM
    
    def initMaxValue(self):# TODO 调用其他的已知算法，求得大约的最优解
        pass
    
    @property
    def reward_pre(self):# deleted old algorithm
        result = 0
        for i in range(self.REWARDRUNTIMES):
            selectVer = self.selectVertex().reshape([self.n,1])
            tempgraph = self.out_array()*self.casRate >= np.random.random([self.n,self.n])
            succeedver = np.sum((selectVer*tempgraph).astype('bool'),0).astype('bool')
            result += np.sum(succeedver)      
        return result/self.REWARDRUNTIMES
    
    @property
    def reward_pre2(self):# TODO Time is too long
        result = 0
        for i in range(self.REWARDRUNTIMES):
            temp = self.graph.copy()
            for edge in self.graph.edges:
                if self.casRate < random.random():
                    temp.remove_edge(edge[0],edge[1])
            tempset = set()          
            runsets = list(nx.connected_components(temp))
            for node in self.selectset:
                for unionset in runsets:
                    if node in unionset:
                        tempset = tempset | unionset
                        break
            result += len(tempset)       
        return result/self.REWARDRUNTIMES            
            
    
    
a = GraphEnv(10,maxSelectNum = 2,MAXN = 11)
a.select_list(range(1))
state,action_onehot,reward,next_state,done = a.act(1)
#a = nx.barabasi_albert_graph(10,5)
#
#temp = a.copy()
#for edge in a.edges:
#    if 0.3 < random.random():
#        temp.remove_edge(edge[0],edge[1])