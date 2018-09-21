#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:23:31 2018

@author: gaoyiwei
"""

import tensorflow as tf

class newattentionModel(object):
    def __init__(self,config,is_training,embeddingl):
        
        #embedding=tf.constant(embeddingl)
        #param
        self.wordEmbeddingSize=config.wordEmbeddingSize
        self.wordsnum=config.wordsnum
        self.wordtotext_hidsize=config.wordtotext_hidsize
        self.keep_prob=config.keep_prob
        self.batchsize=config.batchsize
        self.wssize=config.wssize
        self.classnum=config.classnum
        self.l2=config.l2
        self.learningrate=config.lrre
        #input
        self.xforward=tf.placeholder(tf.int32,[self.batchsize,self.wordsnum])  #batch*words
        self.xbackward=tf.placeholder(tf.int32,[self.batchsize,self.wordsnum])
        self.y=tf.placeholder(tf.int64,[self.batchsize])
        self.maskforward=tf.placeholder(tf.int32,[self.batchsize,self.wordsnum])
        self.maskbackward=tf.placeholder(tf.int32,[self.batchsize,self.wordsnum])
        self.maskpos=tf.placeholder(tf.int32,[self.batchsize])
        
        embedding = tf.get_variable(name="embedding",shape=[self.vocab_size, self.wordEmbeddingSize],initializer=tf.constant_initializer(embeddingl))#,shape=[self.vocab_size, self.embed_size]) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)

        self.inputf=tf.nn.embedding_lookup(embedding,self.xforward) #batch,words,embedding
        self.inputb=tf.nn.embedding_lookup(embedding,self.xbackward)
        
        #self.inputf=tf.nn.dropout(self.inputf,keep_prob=self.keep_prob)
        #self.inputb=tf.nn.dropout(self.inputb,keep_prob=self.keep_prob)        
        
        self.initial_weight()
        #forward
        input2=tf.split(self.inputf,self.wordsnum,1)  #list,words,batch,1,emb 
        
        self.input3=[tf.squeeze(x,[1]) for x in input2] #list,lenth is words,every is batch,emb               
        lstm_cell=tf.contrib.rnn.BasicLSTMCell(self.wordtotext_hidsize)
        if self.keep_prob<1:
            lstm_cell=tf.contrib.rnn.DropoutWrapper(lstm_cell,output_keep_prob=self.keep_prob)
        cell=tf.contrib.rnn.MultiRNNCell([lstm_cell]*1)       
        self.initial_state=cell.zero_state(self.batchsize,tf.float32)  #initial state  t=0 :c and h                   
        self.output=[]
        state=self.initial_state   
        with tf.variable_scope("LSTM_layerf"):
            for time_step,data in enumerate(self.input3):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                cell_output,state=cell(data,state)
                self.output.append(cell_output)
                
        hidden_state_1=tf.stack(self.output,axis=1) #batch,words,wordtosen_hidsize
        
        #backforward
        binput2=tf.split(self.inputb,self.wordsnum,1)  #list,words,batch,1,emb       
        self.binput3=[tf.squeeze(x,[1]) for x in binput2] #list,lenth is words,every is batch,emb        
        blstm_cell=tf.contrib.rnn.BasicLSTMCell(self.wordtotext_hidsize)
        if self.keep_prob<1:
            blstm_cell=tf.contrib.rnn.DropoutWrapper(blstm_cell,output_keep_prob=self.keep_prob)
        bcell=tf.contrib.rnn.MultiRNNCell([blstm_cell]*1)       
        self.binitial_state=bcell.zero_state(self.batchsize,tf.float32)  #initial state  t=0 :c and h                   
        self.boutput=[]
        bstate=self.binitial_state   
        with tf.variable_scope("LSTM_layerb"):
            for time_step,data in enumerate(self.binput3):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                cell_output,bstate=bcell(data,bstate)
                self.boutput.append(cell_output)
                
        bhidden_state_1=tf.stack(self.boutput,axis=1) #batch,words,wordtosen_hidsize
        
        '''  修改一下，取pad之前的最后位置
        forward_last=hidden_state_1[:,self.wordsnum-1,:] #batch,hidsize  hn
        backward_last=bhidden_state_1[:,self.wordsnum-1,:] #batch,hidsize  h1
        '''
        forward_last=[]  #batch,hidsize
        backward_last=[]  #batch,hidsize
        for i in range (self.batchsize):
            forward_last.append(hidden_state_1[i,self.maskpos[i],:])
            backward_last.append(bhidden_state_1[i,self.maskpos[i],:])
        
        
        fbhid_last=tf.concat([forward_last,backward_last],1) #batch,2*hidsize
        
        #here, bhidden_state need to reverse and then concate forward     
        #here is  reverse part of no pad          
        #reverse_bhidden_state=tf.reverse(bhidden_state_1,[1])
        reverse_bhidden_state1=[]
        for i in range(self.batchsize):
            curpos=self.maskpos[i]
            curh=bhidden_state_1[i,0:curpos+1,:]
            b=tf.reverse(curh,[1])
            c=tf.concat([b,bhidden_state_1[i,curpos+1:self.wordsnum,:]],0)
            reverse_bhidden_state1.append(c)
        d=tf.reshape(reverse_bhidden_state1,[self.batchsize*self.wordsnum,self.wordtotext_hidsize])
        reverse_bhidden_state=tf.reshape(d,[self.batchsize,self.wordsnum,self.wordtotext_hidsize])
        
        
        
        
        
        fbhid=tf.concat([hidden_state_1,reverse_bhidden_state],2) #batch,words,2*hidsize
                        
        fbhid_reshape=tf.reshape(fbhid,[-1,self.wordtotext_hidsize*2])
        
        #attention
        a=tf.concat([fbhid_last,fbhid_last],1)
        for i in range(self.wordsnum-2):
            a=tf.concat([a,fbhid_last],1)
        fbhid_last_expand=tf.reshape(a,[self.batchsize,self.wordsnum,self.wordtotext_hidsize*2])
        
        fbhid_last_reshape=tf.reshape(fbhid_last_expand,[-1,self.wordtotext_hidsize*2])
        
        midre1=tf.matmul(fbhid_reshape,tf.transpose(self.ws))
        midre2=tf.matmul(fbhid_last_reshape,tf.transpose(self.us))        
        midre3=midre1+midre2+self.bias
        midre4=tf.nn.tanh(midre3)   #batch*words,wssize
        midre5=tf.multiply(midre4,self.va)#batch*words,wssize
        midre6=tf.reduce_sum(midre5,1)#batch*words
        
        
        fi3=tf.reshape(midre6,[-1,self.wordsnum])#batch,words
        
        rmaxhang=tf.reduce_max(fi3,1,True) #batch
        midd=fi3-rmaxhang  #batch,words   
        

        rpattention=tf.nn.softmax(midd)#batch,words
        
        #here is remove pad softmax
        finalatt=[]
        for i in range(self.batchsize):
            curpos=self.maskpos[i]
            curatt=rpattention[i,0:curpos+1]
            lins=tf.reduce_sum(curatt)
            att=tf.div(curatt,lins)
            lz=tf.zeros([self.wordsnum-1-curpos],dtype=tf.float32)
            lb=tf.concat([att,lz],0)
            finalatt.append(lb)            
        reshapea=tf.reshape(finalatt,[self.batchsize*self.wordsnum])
        rpattention=tf.reshape(reshapea,[self.batchsize,self.wordsnum])
        
        
        rsae=tf.expand_dims(rpattention,2) #batch,words,1
        rdocp=tf.multiply(rsae,fbhid) #batch,words,hid
        self.final_docp=tf.reduce_sum(rdocp,1) #batch,hid
        
        #final mlp
        self.logits=tf.matmul(self.final_docp,self.w1)+self.b1 #bath,classe
        #define loss jiaochashang
        with tf.name_scope("losscost_layer"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)            
            self.cost = tf.reduce_mean(self.loss)
            
            self.l2_loss=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2
            self.cost=self.cost+self.l2_loss
            
        #define accuracy
        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.logits,1)
            correct_prediction = tf.equal(self.prediction,self.y)
            self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")
        if not is_training:
            return
        
        #you hua
        self.global_step = tf.Variable(0,name="global_step",trainable=False)
        #self.lr = tf.Variable(0.0,trainable=False)
        #learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_steps,self.decay_rate, staircase=True)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),config.max_grad_norm)     
        optimizer = tf.train.AdadeltaOptimizer(self.learningrate)
        self.train_op=optimizer.apply_gradients(zip(grads, tvars),global_step=self.global_step)

        #self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
        #self._lr_update = tf.assign(self.lr,self.new_lr)
        
    #def assign_new_lr(self,session,lr_value):
        #session.run(self._lr_update,feed_dict={self.new_lr:lr_value}) 
                            
        
        
    def initial_weight(self):
        self.ws=tf.get_variable("ws",shape=[self.wssize,self.wordtotext_hidsize*2])
        self.us=tf.get_variable("us",shape=[self.wssize,self.wordtotext_hidsize*2])
        self.va=tf.get_variable("va",shape=[self.wssize])
        self.bias=tf.get_variable("bias",shape=[self.wssize])
        self.w1=tf.get_variable("w22",shape=[self.wordtotext_hidsize*2,self.classnum])
        self.b1=tf.get_variable("b22",shape=[self.classnum])
        
        
        
        
        
        
        
        
        
        
