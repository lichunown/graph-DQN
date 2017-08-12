# IM-DQN

### （Influence Maximazation problem with Deep Q-Learning)

使用DQN来解决IM问题，大幅度提升了应用速度。

> **status: working**

## To do list

- [ ] s2v适配
- [ ] s2v with weights
- [ ] DQN

## LOG 

### v0.1 (abandoned)

#### info

经过多方讨论,决定重新修改算法...
#### code

[randomgraph.py](https://github.com/j2kun/erdos-renyi)

marked but not use in this project

### 重难点

- (struct2vec)
- 计算图的覆盖率算法（评估扩散程度、优劣）
- 相似图生成


- 多线程优化（现在暂不考虑）
  cooperative multiagent rl, 多agent,多reward处理

### 一些问题
- action过多, random很难对所有的action遍历, 导致DQN很难random到最优解进行训练.
- 与最大割问题不同, 交换点的这种算法实际与前一个action无关. 即每一个动作之间不存在相互关联, DQN的理论似乎不能套在这里头. 

## v0.2(working)

- 传统的s2v只能将e和v的链接关系（结构）的特征提取出来，而将不包括边和点的属性作为前提条件

## TODO ing

