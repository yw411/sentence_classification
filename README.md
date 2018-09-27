# sentence_classification
针对句子、document层级的分类

句子层级常用数据集：SST、MR、Subj

常用方法：

1）cnn

多个通道、多个过滤器、word和char相结合

2）lstm、attention

tree-lstm较多见

3)cnn-lstm

先提取短语信息，再lstm或者先lstm，再cnn(我比较倾向于前者)

4）额外信息

主要是加入情感词典信息

5）多任务multi-task

这个方法也发表过不少

### CNN

1、kim  Convolutional Neural Networks for Sentence Classification



### 结果

Model | SST-2  | SST-5 | MR  |
