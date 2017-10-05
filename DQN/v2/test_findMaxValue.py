# -*- coding: utf-8 -*-
from graph import GraphEnv
import numpy as np

MAXVERTEXNUM = 50

env = GraphEnv(50,5,MAXVERTEXNUM)
env.load('test')

env.initMaxValue()

print('[DONE]    maxValue: {}'.format(env.maxValue))

np.save('maxValue.selectVertex',env.maxValue_Vertex)