#coding:utf-8
import tensorflow as tf
import numpy as np
import ELMO as Model
import utils
import Config
import matplotlib.pyplot as plt

        
    
config = Config.Big5Config()
config.keep_prob = 1.0
config.batch_size = 5
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1
data = utils.DataStream( utils.answers, utils.features, config, config.batch_size, need_random="none")

   
list_forwards = []   
list_backwards = []  
list_scores = []  
arr = np.arange(19718)

with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                            config.init_scale)
    
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        mtest = Model.Model(is_training=False, config=config)
    #tf.global_variables_initializer().run()  
    model_saver = tf.train.Saver()
    print ('model loading ...')
    model_saver.restore(session, 'data/BIG5/model/lm-4')
    print ('Done!')

    
    for ids in range(int(15000/config.batch_size),int(19718/config.batch_size)):
        if ids%200==0:
            print(ids,end=",")
        a,b,c = data.oppo_feed_dic(ids*config.batch_size,arr)
        feed_dicts = mtest.create_feed_dict(a,b,c)
        elmof_acc,elmob_acc,accuracy_class= session.run([
                                                        mtest.f_accuracy,
                                                        mtest.b_accuracy,
                                                        mtest.accuracy_class],
                                                        feed_dicts)
        
        list_forwards.append(elmof_acc)
        list_backwards.append(elmob_acc)
        list_scores.append(accuracy_class)
        
        
print(np.mean(list_forwards))
print(np.mean(list_backwards))
print(np.mean(list_scores))
        
#f = []
#b = []
#s = []
#            
#for user in range(19718):
#    f_score = sum(list_forwards[user*49:user*49+49])
#    b_score = sum(list_backwards[user*49:user*49+49])
#    score = f_score +b_score
#    f.append(f_score)
#    b.append(b_score)
#    s.append(score)
#    
#    
#
#pros = np.array(s)
#plt.figure(figsize=(10, 12))
#plt.hist(pros,
#         bins=500,
#         normed=1,
#         facecolor="red",
#         edgecolor="red",
#         alpha=0.7)
#plt.savefig('tt.png')
#plt.show()
#
#for i,x in enumerate(s):
#    if x>160:
#        print(i,x,utils.answers[i])
# 
#t = 0
#for i,x in enumerate(utils.answers):
#    if sum(utils.answers[i])==150:
#        t+=1
#        print(i,x,utils.answers[i])
#        
#        
