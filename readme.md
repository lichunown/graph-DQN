# **IM-DQN**  *(Influence Maximization problem with Deep Q-Network)*


**`status: working`**

使用DQN来解决IM问题，大幅度提升了速度。

## To-Dos

- [ ] s2v适配
- [ ] s2v with weights
- [ ] DQN

## LOGs 

### v0.2 (working)

#### To do

- 传统的s2v只能将e和v的链接关系（结构）的特征提取出来，而将不包括边和点的属性作为前提条件
- 现阶段针对较为特殊的固定传播概率的问题进行讨论来验证DQN部分的正确性，使得所有的边权为一个恶固定的值，较为简单，使用现有s2v可以解决。

#### Key points

#### Changed

#### Problems Review

### v0.1 (abandoned)

#### To do

#### Key Points

- (struct2vec)
- 计算图的覆盖率算法（评估扩散程度、优劣）
- 相似图生成


- 多线程优化（现在暂不考虑）
  cooperative multiagent rl, 多agent,多reward处理
- 替换选择的节点来进行迭代

#### Changed

经过讨论，重新修改算法，此版本废弃。

#### Problems Review
- action过多, random很难对所有的action遍历, 导致DQN很难random到最优解进行训练.
- 与最大割问题不同, 交换点的这种算法实际与前一个action无关. 即每一个动作之间不存在相互关联, DQN的理论似乎不能套在这里头.使用逐个加点的方式进行迭代来改进。

## Refrences

- [1]	J. Lee, H. Kim, J. Lee, and S. Yoon, “Intrinsic Geometric Information Transfer Learning on Multiple Graph-Structured Datasets,” arXiv, vol. cs.NE. 15-Nov-2016.
- [2]Y. Tang, Y. Shi, and X. Xiao, “Influence Maximization in Near-Linear Time,” presented at the the 2015 ACM SIGMOD International Conference, New York, New York, USA, 2015, pp. 1539–1554.
- [3]L. F. R. Ribeiro, P. H. P. Saverese, and D. R. Figueiredo, “struc2vec,” presented at the the 23rd ACM SIGKDD International Conference, New York, New York, USA, 2017, pp. 385–394.
- [4]H. Dai, B. Dai, and Le Song, “Discriminative Embeddings of Latent Variable Models for Structured Data,” arXiv. pp. 1–23, 28-Sep-2016.
- [5]W. Chen, Y. Wang, and S. Yang, “Efficient Influence Maximization in Social Networks,” presented at the KDD '09, 2009, pp. 1–9.
- [6]Y. Chen, E. Matus, and G. P. Fettweis, “Combined TDM and SDM Circuit Switching NoCs with Dedicated Connection Allocator,” presented at the 2017 IEEE Computer Society Annual Symposium on VLSI (ISVLSI, 2017, pp. 104–109.
- [7]H. Dai, E. B. Khalil, Y. Zhang, B. Dilkina, and Le Song, “Learning Combinatorial Optimization Algorithms over Graphs,” arXiv. pp. 1–28, 04-May-2017.
- [8]A. Phadtare, A. Bahmani, A. Shah, and R. Pietrobon, “Scientific writing: a randomized controlled trial comparing standard and on-line instruction,” BMC Medical Education, vol. 9, no. 1, p. 27, May 2009.

