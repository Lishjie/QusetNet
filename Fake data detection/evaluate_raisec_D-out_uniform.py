"""
@author: lishijie
@date: 2019/12/04
@description: 使用WGAN-GP构建语言模型
              raise心理问卷作为真实数据集
"""
import matplotlib.pyplot as plt
import tflib.ops.conv1d
import tflib.ops.linear
import tflib as lib
import language_helpers
import tensorflow as tf
import numpy as np
import datetime
import time
import os
import sys
# 添加当前的工作路径到系统路径中
# os.getcwd() 获取当前的工作路径
sys.path.append(os.getcwd())


# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!
DATA_DIR = './save/riasec/fake_data'
if len(DATA_DIR) == 0:
    raise Exception(
        'Please specify path to data directory in gan_language.py!')

BATCH_SIZE = 128  # Batch size
ITERS = 10000  # How many iterations to train for
SEQ_LEN = 48  # Sequence length in characters
DIM = 20   # Model dimensionality. This is fairly slow and overfits, even on
# Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 10  # How many critic iterations per generator iteration. We
# use 10 for the results in the paper, but 5 should work fine
# as well.
LAMBDA = 10  # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 9900  # Max number of data examples to load. If data loading 7187
# is too slow or takes too much RAM, you can decrease
# this (at the expense of having less training data).

use_true_data = False
if use_true_data:
    FILE_PATH = './save/riasec/real_data.txt'   #
else:
    FILE_PATH = './save/riasec/samples/samples_100.txt'
# 以字典的形式返回当前位置的全部局部变量
# print_model_settings - 打印输入字典的值
lib.print_model_settings(locals().copy())

# lines: 训练数据
# charmap: 词汇表（包含索引）
# inv_charmap: 所有的词汇（也就是charmap中的key）


#tokenize:True 表示之间有空格，False表示无空格
lines, charmap, inv_charmap = language_helpers.load_dataset_from_file(
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    tokenize=use_true_data,
    max_vocab_size=6,
    data_dir=FILE_PATH,
)
# 定义softmax
# logits -> batch_size * sequence_length * vocab_size
# return -> batch_size * sequence_length * vocab_size


def softmax(logits):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, len(charmap)])
        ),
        tf.shape(logits)
    )

# 添加噪声数据（从均匀噪声中进行随机采样sample）


def make_noise(shape):
    return tf.random_uniform(shape)


"""
@description: 残差卷积神经网络(ResNet)
@argvs: name - 残差卷积神经网络的名字
        inputs - 网络的输入
"""


def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    # input shape -> batch_size * dim * sequence_length
    # output shape -> batch_size * dim * sequence_length
    output = lib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 10, output)
    output = tf.nn.relu(output)
    # input -> batch_size * dim * sequence_length
    # output -> batch_size * dim * sequence_length
    output = lib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 10, output)
    # return -> batch_size * dim * sequence_length
    return inputs + (0.3*output)


"""
@desciption: 生成器(Generator)，生成器逐步将高斯噪声数据拟合成
             和真实数据一样分布的数据
@argvs: n_samples - 采样个数 batch_size
        prev_outputs -
"""


def Generator(n_samples, prev_outputs=None):
    output = make_noise(shape=[n_samples, 128])
    # input shape -> n_samples * 128
    # output shape -> n_samples * (SEQ_LEN*DIM)
    output = lib.ops.linear.Linear('Generator.Input', 128, SEQ_LEN*DIM, output)
    # output -> batch_size * dim * sequence_length
    output = tf.reshape(output, [-1, DIM, SEQ_LEN])
    # 经过5层残差卷积神经网络(ResNet)
    # input -> batch_size * dim * sequence_length
    output = ResBlock('Generator.1', output)
    output = ResBlock('Generator.2', output)
    output = ResBlock('Generator.3', output)
    output = ResBlock('Generator.4', output)
    output = ResBlock('Generator.5', output)
    # input -> batch_size * dim * seuqence_length
    # output -> batch_size * vocab_size * sequence_length
    output = lib.ops.conv1d.Conv1D(
        'Generator.Output', DIM, len(charmap), 1, output)
    # batch_size * sequence_length * vocab_size
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output)
    # return batch_size * sequence_length * vocab_size
    return output


"""
@description: 判别器(discriminator) 用于拟合样本的Wasserstein距离
              input_size -> batch_size * sequence_length * vocab_size
              output_size -> batch_size * 1
"""


def Discriminator(inputs):
    # output -> batch_size * vocab_size * sequence_length
    output = tf.transpose(inputs, [0, 2, 1])
    # output -> batch_size * 5 * sequence_length
    output = lib.ops.conv1d.Conv1D(
        'Discriminator.Input', len(charmap), DIM, 1, output)
    output = ResBlock('Discriminator.1', output)
    output = ResBlock('Discriminator.2', output)
    output = ResBlock('Discriminator.3', output)
    output = ResBlock('Discriminator.4', output)
    output = ResBlock('Discriminator.5', output)
    # output -> batch * (DIM*SEU_LEN)
    output = tf.reshape(output, [-1, SEQ_LEN*DIM])
    # output -> batch_size * 1
    output = lib.ops.linear.Linear(
        'Discriminator.Output', SEQ_LEN*DIM, 1, output)
    return output


# real data
real_inputs_discrete = tf.placeholder(
    tf.int32, shape=[MAX_N_EXAMPLES, SEQ_LEN])
# real_inputs -> batch_size * sequence_length * vocab_size
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
# fake_inputs -> batch_size * sequence_length * vocab_size
fake_inputs = Generator(BATCH_SIZE)
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

# 利用Discriminator求解出样本的Wassertein距离
disc_real = Discriminator(real_inputs)

# disc_fake = Discriminator(fake_inputs)
#
# # 判别器(Discriminator)和生成器(Generator)的损失函数
# # 判别器(Discriminator)的作用是拟合出样本之间的距离
# disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
# # 生成器(Generator)的作用是缩小wassertein距离
# gen_cost = -tf.reduce_mean(disc_fake)
#
# # WGAN lipschitz-penalty
# alpha = tf.random_uniform(
#     shape=[BATCH_SIZE,1,1],
#     minval=0.,
#     maxval=1.
# )
# # 真实样本和生成样本之间的向量差距
# differences = fake_inputs - real_inputs
# interpolates = real_inputs + (alpha*differences)


# Dataset iterator
# 构建真实样本迭代器
def inf_train_gen():
    while True:
        # lines为所有的训练数据
        #np.random.shuffle(lines)
        np.array(lines)
        for i in range(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
            # yield相当构建一个迭代器
            # 程序每次调用next方法之后迭代器会进行一次迭代，之后便终止运行等待下一次迭代
            # 将获取的文本信息转化为index信息
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]],
                dtype='int32'
            )


# 在正式开始训练模型之前，本实验先检测前4元语言模型对于真实数据和生成数据之间的 JS 距离
# 通过评判这些模型的拟合效果来选择n的值
# 目的是检测出 几元语言模型 对于样本的分布的表征更好
# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
true_char_ngram_lms = [language_helpers.NgramLanguageModel(
    i+1, lines[10*BATCH_SIZE:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [language_helpers.NgramLanguageModel(
    i+1, lines[:10*BATCH_SIZE], tokenize=False) for i in range(4)]
for i in range(4):
    print("validation set JSD for n={}: {}".format(
        i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
true_char_ngram_lms = [language_helpers.NgramLanguageModel(
    i+1, lines, tokenize=False) for i in range(4)]

# 正式开始WGAN-GP的测试过程
which_D = 9000  # 指定使用哪一轮的模型来进行训练
#checkpoint_file_no_GCN = 'save/riasec/model/WGAN-model-'+str(which_D)
model_path = input("model_path:")
checkpoint_file_no_GCN = model_path+str(which_D)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
gen = inf_train_gen()
_data = gen.__next__()
#print(_data)

# 路径配置
true_data_path = input("true data path:")
true_data_result = input("true data test result:")
fake_data_path = input("Fake Data path:")
fake_data_result = input("Fake Data test result:")

#[250,300,350,400,450,500,600,700,800,900,1000]

# W距离的存储位置
try:
    result_save_path = input("result_save_path")
    os.mkdir(result_save_path+str(which_D))
except:
    pass

j = 0
with tf.Session() as sess:
    sess.run(init)
    # saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file_no_GCN))
    saver.restore(sess, checkpoint_file_no_GCN)
    print('finish loading model!')

    use_true_data = True
    if use_true_data:
        #FILE_PATH = './save/riasec/real_data.txt'  #
#        FILE_PATH = './save/riasec/testData_8200/real.txt'
        FILE_PATH = true_data_path
    else:
        FILE_PATH = './save/riasec/samples/samples.txt'

    lines, charmap, inv_charmap = language_helpers.load_dataset_from_file(
        max_length=SEQ_LEN,
        max_n_examples=MAX_N_EXAMPLES,
        tokenize=True,
        max_vocab_size=6,
        data_dir=FILE_PATH,
    )

    lines_array = np.array(
        [[charmap[c] for c in l] for l in lines],
        dtype='int32')

    D_real = sess.run(disc_real, feed_dict={real_inputs_discrete: lines_array})

    #GAN_D out of fake data
    # f1 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150,
    #       200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1500, 2000]
    # f2 = [i for i in range(2500, 50000, 100)]

#    f1 = [i for i in range(0, 10001, 50)] + [i for i in range(10000, 50000, 1000)]

    # FILES = f1+f2
#    FILES = f1
    #plt.figure(21)
#    np.savetxt('./save/riasec/model_Dout/D_out' +
    np.savetxt(true_data_result +
               str(which_D)+'/real.txt', D_real)
#    files = os.listdir("./save/riasec/testData_8200/")
    files = os.listdir(fake_data_path)
    for file in files:
        if file == "real.txt":
            continue
#    for i in FILES:
#        use_true_data = False
#        if use_true_data:
#            FILE_PATH = './save/riasec/real_data.txt'  #
#        else:
#            FILE_PATH = './save/riasec/samples/samples_' + str(i) + '.txt'

#        FILE_PATH  = "./save/riasec/testData_8200/"+file
        FILE_PATH  = fake_data_path+file
        lines, charmap, inv_charmap = language_helpers.load_dataset_from_file(
            max_length=SEQ_LEN,
            max_n_examples=MAX_N_EXAMPLES,
            tokenize=True,
            max_vocab_size=6,
            data_dir=FILE_PATH,
        )

        lines_array = np.array(
            [[charmap[c] for c in l] for l in lines],
            dtype='int32')

        D_fake = sess.run(disc_real, feed_dict={
                          real_inputs_discrete: lines_array})
        print('--load model form ', checkpoint_file_no_GCN)
        print('--load data from ', FILE_PATH)
#        print(D_fake)
#        savetxt_dir = './save/riasec/model_Dout/D_out8200/D_out' + \
        savetxt_dir = fake_data_result + \
            file.split(".")[0] + '.txt'
        np.savetxt(savetxt_dir, D_fake)
        print('--save in ', savetxt_dir, '\n')

#        Y = np.random.uniform(0, 10, 640)

#
#        #plt.subplot(4,4,1 + j)
#
#        plt.figure(figsize=(10,10))
#
#        plt.scatter(D_fake[:,0], Y, c='b', marker='o')
#        plt.scatter(D_real, Y, c='r', marker='o')
#        plt.xlim((-12, 12))
#        plt.xticks([])  # ignore xticks
#        plt.ylim((0, 10))
#        plt.yticks([])  # ignore yticks
#        plt.savefig('./save//riasec/D_plot'+str(which_D)+ '/' + str(i) + '.jpg')
#
#        j += 1
    # # plt.show()
    # FILES = [1500,2000,2500,3500,4000,10000]
    # # FILES = [3000, 4000, 10000,40000]
    # # plt.figure(21)
    #
    # for i in FILES:
    #     use_true_data = False
    #     if use_true_data:
    #         FILE_PATH = './save/riasec/real_data.txt'  #
    #     else:
    #         FILE_PATH = './save/riasec/samples/samples_' + str(i) + '.txt'
    #
    #     lines, charmap, inv_charmap = language_helpers.load_dataset_from_file(
    #         max_length=SEQ_LEN,
    #         max_n_examples=MAX_N_EXAMPLES,
    #         tokenize=False,
    #         max_vocab_size=6,
    #         data_dir=FILE_PATH,
    #     )
    #
    #     lines_array = np.array(
    #         [[charmap[c] for c in l] for l in lines],
    #         dtype='int32')
    #
    #     D_fake = sess.run(disc_real, feed_dict={real_inputs_discrete: lines_array})
    #     print(' the dis of fake sata:')
    #     print(D_fake)
    #
    #     Y = np.random.uniform(0, 10, 640)
    #
    #     # plt.subplot(4,4,1 + j)
    #
    #     plt.figure(figsize=(10, 10))
    #
    #     plt.scatter(D_fake[:, 0], Y, c='b', marker='o')
    #     plt.scatter(D_real, Y, c='r', marker='o')
    #     plt.xlim((-12, 12))
    #     plt.xticks([])  # ignore xticks
    #     plt.ylim((0, 10))
    #     plt.yticks([])  # ignore yticks
    #     plt.savefig('./save//riasec/D_plot'+str(which_D)+ '/' + str(i) + '.jpg')
