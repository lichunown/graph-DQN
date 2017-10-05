cmake_minimum_required(VERSION 3.6)
project(IMM_SIGMOD15_xxk)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    nethept/attribute.txt
    nethept/graph.txt
    nethept/graph_cont.inf
    nethept/graph_ic.inf
    nethept/graph_lt.inf
    src/sfmt/a.c
    src/sfmt/a.cpp
    src/sfmt/SFMT-alti.h
    src/sfmt/SFMT-common.h
    src/sfmt/SFMT-params.h
    src/sfmt/SFMT-params11213.h
    src/sfmt/SFMT-params1279.h
    src/sfmt/SFMT-params132049.h
    src/sfmt/SFMT-params19937.h
    src/sfmt/SFMT-params216091.h
    src/sfmt/SFMT-params2281.h
    src/sfmt/SFMT-params4253.h
    src/sfmt/SFMT-params44497.h
    src/sfmt/SFMT-params607.h
    src/sfmt/SFMT-params86243.h
    src/sfmt/SFMT-sse2.h
    src/sfmt/SFMT.c
    src/sfmt/SFMT.h
    src/sfmt/test.c
    src/continuous_rrset.h
    src/discrete_rrset.h
    src/graph.h
    src/head.h
    src/iheap.h
    src/imm.cpp
    src/imm.h
    src/infgraph.h
    src/weibull.h
    imm_continuous
    imm_discrete
    Makefile)

add_executable(IMM_SIGMOD15_xxk ${SOURCE_FILES})