# sentence_classification

## 定义
针对句子层级sentence-level的分类

句子层级常用数据集：SST、MR、Subj、imdb、trec

情感分类有很多子课题可以研究，句子级别、文本级别、aspect级别、cross-domain、cross-language，观点提取（semeval评测会议），立场检测（方法差不多）、构建情感词典、构建情感词向量

## 发展脉络

1）cnn

多个通道、多个过滤器、word和char相结合，dynamic-k-max-pooling

2）lstm、attention

tree-lstm较多见，和情感信息结合比较常见

3)cnn-lstm

先提取短语信息，再lstm或者先lstm，再cnn(我比较倾向于前者)

4）额外信息

主要是加入情感词典信息

5）多任务multi-task

这个方法也发表过不少

6)递归神经网络

### CNN

1、kim  Convolutional Neural Networks for Sentence Classification

2、A Convolutional Neural Network for Modelling Sentences

作者加入dynamic-k-pooling ，k是根据句子长度和网络深度动态变化的，k-pooling是在一个序列q中选取k个最大的，而不是一个最大的。作者同时使用宽卷积

3、Multichannel Variable-Size Convolution for Sentence Classification

多通道，不同的初始化方式，dynamic-k，

4、MGNC-CNN: A Simple Approach to Exploiting Multiple Word Embedding

同样是多个通道，但是是异构方式，每个embedding上分别进行，最后concat，不同于3

5、Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts

字向量和词向量结合。

### cnn-rnn

6、Recurrent Convolutional Neural Networks for Text Classification   2015  最早的一篇

还不是简单的rnn，作者自己发明了一套公式捕捉上下文信息，然后是1维cnn

7、Dependency Sensitive Convolutional Neural Networks for Modeling Sentences and Documents

先lstm再cnn，（其实我一直不认可这种顺序）

8、Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling

同6，只是最后用了2d的cnn，当做图像来处理，没法说。

### tree、递归神经网络  （没怎么关注，因为要用到短语树）

9、Long Short-Term Memory Over Tree Structures

tree-lstm  当前节点受到子节点的影响

10、Improved Semantic Representations FromTree-Structured Long Short-Term Memory Networks

tree-lstm

11、When Are Tree Structures Necessary for Deep Learning of Representations?

12、socher多篇文章

### multi-task方法

13、Recurrent Neural Network for Text Classification with Multi-Task Learning

多个任务在这里指多个数据集，每个数据集上是一个任务，是个什么原理呢

13.1 A Generalized Recurrent Neural Architecture for Text Classification with Multi-Task Learning

在13的基础上进行的改进

### 嵌入情感词信息

14、Context-Sensitive Lexicon Features for Neural Sentiment Analysis

情感词典中的词标记出来，通过hidden state得到一个权重，一个句子中所有情感词加权有一个得分，句子表示最后有一个得分，两个得分加权得到最后的分数

15、Linguistically Regularized LSTM for Sentiment Classification

考虑了情感词、否定词、程度副词

16、Sentiment Lexicon Enhanced Attention-Based LSTM for Sentiment Classification  aaai2018 abstract  lei是作者

这是个2页的摘要型文章，引入情感词典，将一个句子中每个情感词的hidden求和平均得到hs，然后每个词与hs进行attention得到最后句子表示。

17、A Multi-sentiment-resource Enhanced Attention Network for Sentiment  作者lei同时发表了好几篇文章    acl2018

利用到了情感词、程度副词、否定词信息，先进行交互（句子和情感词、否定、程度副词），更新表示，再gru，然后同16，计算attention，方法是很新，但是怎么理解呢

18、SAAN: A Sentiment-Aware Attention Network for Sentiment Analysis     sigir2018

公式很复杂感觉。

19、LAAN: A Linguistic-Aware Attention Network for Sentiment Analysis      www2018

和SAAN有点像，也没有搞懂。

20、Lexicon integrated cnn models with attention for sentiment analysis

每个词有一个情感得分，然后映射为向量，3种方式：直接连在词向量后面，2个通道，分别计算cnn（异构）

21、A Lexicon-Based Supervised Attention Model for Neural Sentiment Analysis

在loss上有所改进，

## 结果

Model | SST-5  | SST-2 | MR  | subj |trec
------| -------| ------| ----| -----|----
1     |  48.0  | 88.1  | 81.5| 93.4 | 93.6 
2     |  48.5  | 86.8  | /   | /    | 93.0 
3     |  49.6  | 89.4  | /   | 93.9 | /
4     |48.65   | 88.35 | /   |94.11 | 95.52
5     |48.3    |  85.7 | /   |/     |/
6     |48.7    |  /    | /   |/     | / 
7     |50.6    | 89.1  |82.2 |93.9  | 95.6
8     |52.4    | 89.5  |82.3 |94.0  | 96.1
9     |48      | 81.9  |/    |/     | /
10    |51.0    | 88.0  | /   |/     |/
14    |51.1    | 89.2  | /   |/     |/
15    |50.6    |   /   | 82.1|/     |/
16    |49.3    |       | 84  |/     |/
17    |51.4    |       | 84.5|/     |/
18    |49.7    |       | 84.3|/     |/
19    |49.1    |       | 83.9|/     |/
20    |48.8    |/      |/    |/     |/

## 个人看法

情感分类不同于文本分类，两者有共通之处，比如体育用词多就属于体育，正面词多更倾向于正面情感，但是这样用简单的规则即可，大家研究各种各样的方法，是为了解决那些比较难判断的，或者表述很复杂，不直接表达情感，说这个好，那个不好，但是总体上好。情感分类只用n-gram信息还远远不够，还需要情感信息，比如人们研究如何构造情感词典，使用的特征包括正面情感词的个数、负面的个数、感叹词的个数、否定词的个数等等。

所以当今我认为还是要引入一些额外信息、知识比较好，情感词典，否定词，程度副词，（依存句法，词性可以吗）

所以在神经网络里，我觉得还是要嵌入情感信息，但是该怎么嵌入呢，还没有很好的解决方案。
