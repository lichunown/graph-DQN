#encoding:utf-8

import os
import numpy as np

def getTempFileNum():
    num = 0
    if not os.path.exists('temp/'):
        os.mkdir('temp')
        return 0
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
    print(n,m)
    result = np.zeros([int(n),int(m)])
    for i,H in enumerate(f):
#        print(i,H)
        result[i,:] = np.array([float(num) for num in H.split()[1:]])
    f.close()
    return result

def s2v(edges,s2vlength = 100):
    tempnum = getTempFileNum()
    writeEdges(edges,'temp/%d.edges'%tempnum)
    os.system('./runs2v.sh temp/%d.edges temp/%d.s2v %d'%(tempnum,tempnum,s2vlength))
    return readVec('temp/%d.s2v'%tempnum)
    

