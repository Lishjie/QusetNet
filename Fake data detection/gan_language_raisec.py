"""
@author: lishijie
@date: 2019/12/04
@description: 使用WGAN-GP构建语言模型
              RASIEC理问卷作为真实数据集
1214:training the GAN only uses the train data. 
"""
import tflib.plot
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
DATA_DIR = './save/raisec/'
if len(DATA_DIR) == 0:
    raise Exception(
        'Please specify path to data directory in gan_language.py!')

BATCH_SIZE = 128  # Batch size
ITERS = 50000  # How many iterations to train for
SEQ_LEN = 48  # Sequence length in characters
DIM = 20   # Model dimensionality. This is fairly slow and overfits, even on
# Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 10  # How many critic iterations per generator iteration. We
# use 10 for the results in the paper, but 5 should work fine
# as well.
LAMBDA = 10  # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 70000  # Max number of data examples to load. If data loading
# is too slow or takes too much RAM, you can decrease
# this (at the expense of having less training data).
FILE_PATH = './save/riasec/'

# 以字典的形式返回当前位置的全部局部变量
# print_model_settings - 打印输入字典的值
lib.print_model_settings(locals().copy())

# lines: 训练数据
# charmap: 词汇表（包含索引）
# inv_charmap: 所有的词汇（也就是charmap中的key）
lines, charmap, inv_charmap = language_helpers.load_dataset_realtrain(
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    tokenize=True,
    max_vocab_size=6,
    data_dir=FILE_PATH,
)


def ite_everysave(a):
    if a <= 600:
        b = 50
    if a > 600 and a <= 1000:
        b = 50
    if a > 1000 and a <= 5000:
        b = 50
    if a > 5000 and a <= 10000:
        b = 50
    if a > 10000:
        b = 1000
    return b


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

# 添加噪声数据（从高斯噪声中进行随机采样sample）


def make_noise(shape):
    return tf.random_normal(shape)


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
real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
# real_inputs -> batch_size * sequence_length * vocab_size
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
# fake_inputs -> batch_size * sequence_length * vocab_size
fake_inputs = Generator(BATCH_SIZE)
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

# 利用Discriminator求解出样本的Wassertein距离
disc_real = Discriminator(real_inputs)
disc_fake = Discriminator(fake_inputs)

# 判别器(Discriminator)和生成器(Generator)的损失函数
# 判别器(Discriminator)的作用是拟合出样本之间的距离
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
# 生成器(Generator)的作用是缩小wassertein距离
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN lipschitz-penalty
alpha = tf.random_uniform(
    shape=[BATCH_SIZE, 1, 1],
    minval=0.,
    maxval=1.
)
# 真实样本和生成样本之间的向量差距
differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha*differences)
# 获取梯度信息
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

# 获取生成器(Generator)和判别器(Discriminator)的所有参数
gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

# 进行生成器(Generator)和判别器(Discriminator)的反传过程
gen_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

# Dataset iterator
# 构建真实样本迭代器


def inf_train_gen():
    while True:
        # lines为所有的训练数据
        np.random.shuffle(lines)
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

# 正式开始WGAN-GP的训练过程
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    # 利用Generator来生成一批数据
    def generate_samples():
        # batch_size * sequence_length * vocab_size
        samples = session.run(fake_inputs)
        # batch_size * sequence_length
        samples = np.argmax(samples, axis=2)
        # 将index转化为真实的单词
        decoded_samples = []
        for i in range(len(samples)):
            decoded = []
            for j in range(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
                decoded.append(' ')
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    # 创建一个获取真实数据的迭代器
    gen = inf_train_gen()

    # 打开log文件
    log = open(FILE_PATH+'experiment-log.txt', 'w')
    print('Time: ', datetime.datetime.now())
    print("Start training WGAN-GP ... ")
    buffer = "Time: {}\n".format(datetime.datetime.now())
    log.write(buffer)
    log.write("Start training WGAN-GP ... \n")

    # 控制训练的总轮数
    for iteration in range(ITERS):
        # 记录总体的训练时长
        start_time = time.time()

        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        # Train discriminator
        for i in range(CRITIC_ITERS):
            _data = gen.__next__()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_inputs_discrete: _data}
            )

        # 绘制训练的进度图像
        lib.plot.plot(FILE_PATH+'plot/time', time.time() - start_time)
        lib.plot.plot(FILE_PATH+'plot/train disc cost', _disc_cost)
        buffer = "time: {} train disc cost: {}\n".format(
            datetime.datetime.now(), _disc_cost)
        log.write(buffer)
        save_every = ite_everysave(iteration)
        if iteration % save_every == 0 or iteration == ITERS - 1:
            samples = []
            for i in range(100):
                samples.extend(generate_samples()) 
            for i in range(4):
                lm = language_helpers.NgramLanguageModel(
                    i+1, samples, tokenize=False)
                lib.plot.plot(FILE_PATH+'plot/js{}'.format(i+1),
                              lm.js_with(true_char_ngram_lms[i]))
                buffer = 'js{} {}\n'.format(
                    i+1, lm.js_with(true_char_ngram_lms[i]))
                log.write(buffer)

            # 保存已经生成的文本数据
            with open(FILE_PATH+'samples/samples_{}.txt'.format(iteration), 'w') as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")

        
            # 保存当前的模型
            print('model saving ... ')
            model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            model_saver.save(session, FILE_PATH +
                             'model/WGAN-model-%d' % (iteration))
            print('Done!')

        if iteration % 500 == 0:
            lib.plot.flush()

        lib.plot.tick()
    log.close()
