#coding:utf-8
import tensorflow as tf
import numpy as np
import time
import Config
import ELMO as Model
import utils

config = Config.Big5Config()
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1
data = utils.DataStream( utils.answers, utils.features, config, config.batch_size, need_random="none")

def run_epoch(session, m):
    """Runs the model on the given data."""
    epoch_size = 5749//m.batch_size
    start_time = time.time()
    arr = np.arange(5749)
    np.random.shuffle(arr)
    f_acc = 0.0
    b_acc = 0.0
    c_acc = 0.0
    
    for step in range(epoch_size):
        a,b,c = data.oppo_feed_dic(step*config.batch_size,arr)
        feed_dicts = m.create_feed_dict(a,b,c)
        _,cost,elmof_acc,elmob_acc,accuracy_class= session.run([m.train_op,m.cost,
                                                                m.f_accuracy,
                                                                m.b_accuracy,
                                                                m.accuracy_class],feed_dicts)

        f_acc += elmof_acc
        b_acc += elmob_acc
        c_acc += accuracy_class
        
        if step and step % (epoch_size // 10) == 0:
            print("%.2f cost: %.3f elmof: %.3f elmob: %.3f accuracy_class: %.3f cost-time: %.2f s" %
                (step * 1.0 / epoch_size, cost, f_acc/step,b_acc/step,c_acc/step,
                 (time.time() - start_time)))
            start_time = time.time()
    return cost
    
def main(_):
    
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model.Model(is_training=True, config=config)

        tf.global_variables_initializer().run()
        
        model_saver = tf.train.Saver(tf.global_variables())

        for i in range(config.iteration):
            print("Training Epoch: %d ..." % (i+1))
            train_perplexity = run_epoch(session, m)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            
#            if (i+1) % config.save_freq == 0:
#                print ('model saving ...')
#                model_saver.save(session, config.model_path+'lm-%d'%(i+1))
#                print ('Done!')
            # 保存精度最高的
            
if __name__ == "__main__":
    tf.app.run()
