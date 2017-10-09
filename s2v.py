#encoding:utf-8

import sys,os
import numpy as np

def getTempFileNum():
    num = 0
    tmpfiles = os.listdir('temp/')
    for name in tmpfiles:
        try:
            extmp = int(name.split('.')[0])
        except ValueError:
            extmp = 0
        num = extmp + 1 if num <= extmp  else num
    return num

def writeEdges(edges,filename):
    f = open(filename,'w')
    for edge in edges:
        f.write('%d %d\n'%(edge[0],edge[1]))
    f.close()

def readVec(filename):
    f = open(filename,'r')
    msg = f.readline()
    n,m = msg.split()[0],msg.split()[1]
    result = np.zeros([n,m])
    for i,H in enumerate(f):
        result[i,:] = np.array(H.split())
    f.close()
    return result

def s2v(edges):
    tempnum = getTempFileNum()
    writeEdges(edges,'temp/%d.edges'%tempnum)
    os.system('./runs2v.sh temp/%d.edges temp/%d.s2v'%(tempnum,tempnum))
    return readVec('temp/%d.s2v'%tempnum)
    

