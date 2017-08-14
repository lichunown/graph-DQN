# -*- coding: utf-8 -*-
import numpy as np
import random
from itertools import combinations
from scipy.special import comb
import networkx as nx

class Graph(object):
    def __init__(self,VertexNum = 0,maxVertex=100,enableValue=True):
        '''
        -1: 不存在
         0: 顶点之间未连接
         1: 顶点之间连接
        '''
        if VertexNum>maxVertex:
            raise ValueError('vertex nums({}) is bigger than max vertex nums({}).'.format(VertexNum,maxVertex))
        self.MAXVERTEX = maxVertex
        self.VERTEXNUM = VertexNum
        self.enableValue = enableValue
        self._graph = np.zeros([self.VERTEXNUM,self.VERTEXNUM])
        #self._graph[list(range(self.VERTEXNUM)),list(range(self.VERTEXNUM))] = 0 #
        #self._graph[[[i] for i in range(self.VERTEXNUM)],[list(range(10)) for i in range(self.VERTEXNUM)]] = 0
    
    def addEdge(self,firstV,nextV,directed = True,value = 1):
        if not self.enableValue:
            value = 1
        if firstV>=self.VERTEXNUM or nextV>=self.VERTEXNUM:
            raise KeyError('firstV or nextV is bigger than VERTEXNUM.')
        if directed:
            self._graph[firstV,nextV] = value
        else:
            self._graph[firstV,nextV] = value
            self._graph[nextV,firstV] = value
    def out(self):
        outg = -1*np.ones([self.MAXVERTEX,self.MAXVERTEX])
        temp = self._graph.copy()
        temp[list(range(self.VERTEXNUM)),list(range(self.VERTEXNUM))] = 0
        outg[[[i] for i in range(self.VERTEXNUM)],[list(range(self.VERTEXNUM)) for i in range(self.VERTEXNUM)]] = temp
        return outg
    def isConnected(self,a,b):
        return self.graph[a,b]
    
    def save(self,filename):
        writeGraph = self.graph.copy()
        writeGraph[list(range(self.VERTEXNUM)),list(range(self.VERTEXNUM))] = 0
        np.save(filename,writeGraph)
        
    def load(self,filename):
        self._graph = np.load(filename+'.npy')
        self.VERTEXNUM = len(self.graph)
        
    # TODO readFromMETIS
    
    def random(self,vertexNum=None,edgeGamma=0.5,undirected=True,valuefun = lambda x:1):
        vertexNum = vertexNum if vertexNum else self.VERTEXNUM
        if not undirected:
            self._graph = np.random.random([vertexNum,vertexNum])
            self.VERTEXNUM = vertexNum
            self._graph[np.where(self._graph<edgeGamma)] = 0
            self._graph[np.where(self._graph>=edgeGamma)] = valuefun(len(np.where(self._graph>=edgeGamma)[0]))
        else:
            self._graph = np.zeros([vertexNum,vertexNum])
            for H in range(1,vertexNum):
                for L in range(H):
                    self._graph[H,L] = valuefun(1) if random.random()>=edgeGamma else 0
            for H in range(vertexNum-1):
                for L in range(H+1,vertexNum):
                    self._graph[H,L] = self._graph[L,H]
    
    @property
    def edge(self):
        result = []
        for H in range(self.VERTEXNUM):
            for L in range(self.VERTEXNUM):                
                    result.append((H,L,self.graph[H,L]))
        return result
    
    @property
    def graph(self):
        return self._graph
    
    def toNxDiGraph(self):
        dig = nx.DiGraph()
        for H in range(self.VERTEXNUM):
            for L in range(self.VERTEXNUM):
                if H==L:continue
                if self.graph[H,L] != 0:
                    dig.add_edge(H,L,weight=self.graph[H,L])
        return dig
    
    
    
    def __str__(self):
        return str(self.graph.__str__())
    def __repr__(self):
        return str(self.graph.__repr__())
    def __len__(self):
        return self.VERTEXNUM



class GraphEnv(Graph):
    def __init__(self,VertexNum,selectVerNum,maxVertex=100):
        super(GraphEnv,self).__init__(VertexNum,maxVertex)      
        if selectVerNum>=VertexNum:
            raise ValueError('select vertex nums({}) is bigger than vertex nums({}).'.format(selectVerNum,VertexNum))
        self.SELECTVECNUM = selectVerNum
        self.REWARDRUNTIMES = 800
        #self.initMaxValue()
        self.reset()
        
        
    @property
    def inputSize(self):# 输入action大小
        return self.MAXVERTEX
    
    @property
    def selectVertex(self):# 选中的顶点
        return self.graph[list(range(self.VERTEXNUM)),list(range(self.VERTEXNUM))]
    @selectVertex.setter
    def selectVertex(self,ver):
        self.graph[list(range(self.VERTEXNUM)),list(range(self.VERTEXNUM))] = ver
    
    def selectVertexOut(self):
        r = -1*np.ones([self.MAXVERTEX])
        r[:self.VERTEXNUM] = self.selectVertex
        return r
#    def _selectVertexOut(self):# 为处理神经网络数据产生的
#        result = np.zeros(self.MAXVERTEX)
#        result[list(range(self.VERTEXNUM))] = self.selectVertex
#        return result
    
    def _selectVertexOut_change(self):# 同上，只是在已存在点求了次补
        result = np.zeros(self.MAXVERTEX)
        result[list(range(self.VERTEXNUM))] = 1 - self.selectVertex
        return result
        
    def reset(self):# 重置，为0
        self.selectVertex = 0
        return self.state
    
    def _findMaxIndex(self,r,info):
        return np.where(r*info!=0)[0][np.argmax((r*info)[np.where(r*info!=0)])]
    
    def act(self,action):#action:inputSize  0-(n-1): select new Vertex 
        assert np.sum(self.selectVertex) < self.SELECTVECNUM
        assert len(action) == self.inputSize     
        assert action.shape == (self.MAXVERTEX,)
        #newSelectVec = action.reshape([self.MAXVERTEX])
        #inSelectVec = action[self.MAXVERTEX:2*self.MAXVERTEX]
        newSelect = self._findMaxIndex(action,self._selectVertexOut_change())#outSelectVec*self._selectVertexOut()
        #inSelect = self._findMaxIndex(inSelectVec,self._selectVertexOut_change())#inSelectVec*self._selectVertexOut_change()
        #print(outSelect)
        #outSelect = np.argmax(outSelectVec)
        #inSelect = np.argmax(inSelectVec)
        assert self.graph[newSelect,newSelect] == 0
        #assert self.graph[inSelect,inSelect] == 0
        self.graph[newSelect,newSelect] = 1
        #self.graph[inSelect,inSelect] = 1
        return self.state,self.reward_pre,self.done,self._selectVertexOut_change()
    
    @property
    def state(self):
        return [self.out().reshape([1,self.MAXVERTEX,self.MAXVERTEX]),self.selectVertexOut().reshape([1,self.MAXVERTEX])]

    @property
    def done(self) -> bool:# TODO 如果运行结果比其他算法的最优解好，则done
        return np.sum(self.selectVertex)==self.SELECTVECNUM
    
    def initMaxValue(self):# TODO 调用其他的已知算法，求得大约的最优解
        '''
        自杀式穷举...
        '''
        self.maxValue = 0
        self.maxValue_Vertex = None
        iternum = int(comb(self.VERTEXNUM,self.SELECTVECNUM))
        print('running initMaxValue alliter = %d'%iternum)
        i = 0
        for temp in combinations(range(self.VERTEXNUM),self.SELECTVECNUM):
            self.selectVertex = 0
            self._graph[temp,temp] = 1
            temp_reward_pre = self.reward_pre
            if temp_reward_pre > self.maxValue:
                self.maxValue = temp_reward_pre
                self.maxValue_Vertex = self.selectVertex
            i += 1
            if i%50==0:
                print('running initMaxValue: i:{}/{:d}   maxValue={}'.format(i,iternum,self.maxValue))
        print('running initMaxValue: i:{}/{:d}   maxValue={}'.format(i,iternum,self.maxValue))
        return (self.maxValue,self.maxValue_Vertex)
    @property
    def reward_pre(self):# TODO 算法准确率太低,尝试寻找新的算法解决
        result = 0
        for i in range(self.REWARDRUNTIMES):
            selectVer = self.selectVertex.reshape([self.VERTEXNUM,1])
            tempgraph = self.graph >= np.random.random([self.VERTEXNUM,self.VERTEXNUM])
            succeedver = np.sum((selectVer*tempgraph).astype('bool'),0).astype('bool')
            result += np.sum(succeedver)
        return result/self.REWARDRUNTIMES



#MAXVERTEXNUM = 10
#VERTEXNUM = 10
#SELECTNUM = 3
#assert MAXVERTEXNUM >= VERTEXNUM
#assert VERTEXNUM >= SELECTNUM
#
#a = GraphEnv(VERTEXNUM,SELECTNUM,MAXVERTEXNUM)
##a.random(valuefun = np.random.random)
#a.random(valuefun = lambda x:0.1)
#a.reset()
#
#for i in range(SELECTNUM):
#    action = np.random.random(MAXVERTEXNUM)
#    state,reward_pre,done,info = a.act(action)
#    print('{i}:    {a.selectVertex}    reward1={reward_pre}\n'.format(i=i,a=a,reward_pre=reward_pre,action=action))
#    print(np.max(a._selectVertexOut_change()*action),np.argmax(a._selectVertexOut_change()*action),action[np.argmax(a._selectVertexOut_change()*action)])
#    print(action[np.argmax(a._selectVertexOut_change()*action)]==np.max(a._selectVertexOut_change()*action))
#    if done:
#        break
#
