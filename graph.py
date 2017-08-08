# -*- coding: utf-8 -*-
import numpy as np
import random




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
        outg[[[i] for i in range(self.VERTEXNUM)],[list(range(self.VERTEXNUM)) for i in range(self.VERTEXNUM)]] = self._graph
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
        self.initMaxValue()
        self.reset()
        self.REWARDRUNTIMES = 1000
        
    @property
    def inputSize(self):# 输入action大小
        return 2*self.MAXVERTEX
    
    @property
    def selectVertex(self):# 选中的顶点
        return self.graph[list(range(self.VERTEXNUM)),list(range(self.VERTEXNUM))]
    @selectVertex.setter
    def selectVertex(self,ver):
        self.graph[list(range(self.VERTEXNUM)),list(range(self.VERTEXNUM))] = ver
        
    def _selectVertexOut(self):# 为处理神经网络数据产生的
        result = np.zeros(self.MAXVERTEX)
        result[list(range(self.VERTEXNUM))] = self.selectVertex
        #result[np.where(result==0)]=-9999
        return result
    
    def _selectVertexOut_change(self):# 同上，只是在已存在点求了次补
        result = np.zeros(self.MAXVERTEX)
        result[list(range(self.VERTEXNUM))] = 1 - self.selectVertex
        #result[np.where(result==0)]=-9999
        return result
        
    def reset(self):# 重置，随机产生给定k的选定顶点
        self.selectVertex = 0
        randomsample = random.sample(list(range(self.VERTEXNUM)),self.SELECTVECNUM)
        self.graph[randomsample,randomsample] = 1
        return self.out()
    
    def _findMaxIndex(self,r,info):
        return np.where(r*info!=0)[0][np.argmax((r*info)[np.where(r*info!=0)])]
    
    def act(self,action):#action:inputSize  0-(n-1): out select         n-2n:  in select    [one hot]
        assert len(action) == self.inputSize      
        outSelectVec = action[0:self.MAXVERTEX]
        inSelectVec = action[self.MAXVERTEX:2*self.MAXVERTEX]
        outSelect = self._findMaxIndex(outSelectVec,self._selectVertexOut())#outSelectVec*self._selectVertexOut()
        inSelect = self._findMaxIndex(inSelectVec,self._selectVertexOut_change())#inSelectVec*self._selectVertexOut_change()
        #print(outSelect)
        #outSelect = np.argmax(outSelectVec)
        #inSelect = np.argmax(inSelectVec)
        assert self.graph[outSelect,outSelect] == 1
        assert self.graph[inSelect,inSelect] == 0
        self.graph[outSelect,outSelect] = 0
        self.graph[inSelect,inSelect] = 1
        reward_pre = self.reward_pre(self.REWARDRUNTIMES)
        done = self.done(reward_pre)
        return self.state,reward_pre,done,(self._selectVertexOut(),self._selectVertexOut_change())
    
    @property
    def state(self):
        return self.out()

    def done(self,reward_pre):# TODO 如果运行结果比其他算法的最优解好，则done
        return False
    
    def initMaxValue(self):# TODO 调用其他的已知算法，求得大约的最优解
        pass
    
    def reward_pre(self,n=2000):# TODO 算法准确率太低,尝试寻找新的算法解决
        result = 0
        for i in range(n):
            selectVer = self.selectVertex.reshape([self.VERTEXNUM,1])
            tempgraph = self.graph >= np.random.random([self.VERTEXNUM,self.VERTEXNUM])
            succeedver = np.sum((selectVer*tempgraph).astype('bool'),0).astype('bool')
            result += np.sum(succeedver)
        return result/n





   
#a = GraphEnv(5,3,100)
#a.random(valuefun = np.random.random)
#a.reset()
#for i in range(10):
#    state,reward_pre,done,info = a.act(np.random.random(200))
#    print('{}:    {}    reward1={}    reward2={}'.format(i,a.selectVertex,reward_pre,a.reward_pre()))

