from ctypes import cdll

imm = cdll.LoadLibrary('./imm_discrete.so')

print(imm.main(9,'./imm_discrete -dataset nethept/ -k 50 -model IC -epsilon 0.1'))
