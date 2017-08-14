# -*- coding: utf-8 -*-
from graph import GraphEnv
import numpy as np

MAXVERTEXNUM = 10

env = GraphEnv(10,3,MAXVERTEXNUM)
env.load('test')

env.REWARDRUNTIMES = 10000

env.initMaxValue()

print('[DONE]    maxValue: {}'.format(env.maxValue))

np.save('maxValue.selectVertex',env.maxValue_Vertex)
'''
5.2031
array([ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.])
'''