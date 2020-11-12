"""
@author: lishijie
@data: 2019/12/04
@description: 语言模型库函数
"""
import collections
import numpy as np
import re

# 进行分词
def tokenize_string(sample):
    return tuple(sample.lower().split(' '))

# @author: lishijie
# @description: 用于big5和riasec数据集的分词程序
def tokenize_string_(sample):
    return tuple(sample.strip().split())

def one_by_one_string_(sample):
    return tuple(i for i in sample.strip())
"""
@description: 构建n元语言模型
@argvs: n - 设定n-gram
        samples - 训练数据
        tokenize - 是否需要进行分词处理
"""
class NgramLanguageModel(object):
    def __init__(self, n, samples, tokenize=False):
        # 由于之前获取数据的时候已经经过分词处理了
        # 所以这里一般不需要进行分词处理
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples

        self._n = n
        self._samples = samples
        # 创建一个字典访问器
        # 当访问的key不存在的时候会自动赋值为0
        self._ngram_counts = collections.defaultdict(int)  # 每种模式的Gram的出现次数
        self._total_ngrams = 0                             # 所有的Gram的个数(分母)
        # 开始统计ngram的出现次数
        for ngram in self.ngrams():
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1

    # 从sentences的头部依次获取n个单词
    def ngrams(self):
        n = self._n
        for sample in self._samples:
            for i in range(len(sample)-n+1):
                yield sample[i:i+n]

    # 获取所有语言模型提取出来的模式
    def unique_ngrams(self):
        return set(self._ngram_counts.keys())

    # 计算每种模式在当前语言模型中的出现概率
    def log_likelihood(self, ngram):
        if ngram not in self._ngram_counts:
            return -np.inf
        else:
            return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def kl_to(self, p):
        # p is another NgramLanguageModel
        log_likelihood_ratios = []
        for ngram in p.ngrams():
            log_likelihood_ratios.append(p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        # p is another NgramLanguageModel
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in p.unique_ngrams():
            p_i = np.exp(p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i**2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i**2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, p):
        # p is another NgramLanguageModel
        num = 0.
        denom = 0
        p_ngrams = p.unique_ngrams()
        for ngram in self.unique_ngrams():
            if ngram in p_ngrams:
                num += self._ngram_counts[ngram]
            denom += self._ngram_counts[ngram]
        return float(num) / denom

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    # 用于衡量两个样本之间的JS距离
    def js_with(self, p):
        # 验证集数据的分布情况
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        # 训练集数据的分布情况
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        # 此处根据原始GAN论文P5的公式求得
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        # 计算出 p(x) 和 ( p(x)+q(x)/2 ) 之间的KL距离
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        # 计算出 q(x) 和 ( p(x)+q(x)/2 ) 之间的KL距离
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        # 通过KL距离来计算得出JS距离
        return 0.5*(kl_p_m + kl_q_m) / np.log(2)

"""
加载language model的训练数据
argvs:  max_length - 最大句子长度
        max_n_examples - 训练的数据量
        tokenize -
        max_vocab_size - 词汇量
        data_dir - 数据存放位置
"""
def load_dataset(max_length, max_n_examples, tokenize=False, max_vocab_size=2048, data_dir='/home/ishaan/data/1-billion-word-language-modeling-benchmark-r13output'):
    print("loading dataset...")

    lines = []

    finished = False

    for i in range(99):
        path = data_dir+("/training-monolingual.tokenized.shuffled/news.en-{}-of-00100".format(str(i+1).zfill(5)))
        with open(path, 'r') as f:
            for line in f:
                line = line[:-1]
                if tokenize:
                    line = tokenize_string(line)
                else:
                    line = tuple(line)

                # 截断超长文本
                if len(line) > max_length:
                    line = line[:max_length]

                # 将长度不够的sentences补充字符 '`'
                lines.append(line + ( ("`",)*(max_length-len(line)) ) )

                # 获取指定数量的sentences作为训练数据
                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break

    np.random.shuffle(lines)

    counts = collections.Counter(char for line in lines for char in line)

    # 构建训练语料库的词汇表
    # word: index
    # 未知单词索引值为0
    charmap = {'unk':0}
    inv_charmap = ['unk']

    # 计算所有vocabulary的出现次数，most_common函数会自动排序
    # 排序较为靠后的单词（生僻字）可能会被认为是未知单词
    # 出现次数较多的单词具有较小的索引值
    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    # 将词汇表中不存在的单词映射为位置单词
    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    # 打印获取的前100条sentences的信息
    for i in range(100):
        print(filtered_lines[i])

    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap

"""
@description: 用于BIG5和RAISEC数据集的训练数据读取
"""
def load_dataset_(max_length, max_n_examples, tokenize, max_vocab_size, data_dir):
    print("loading dataset...")

    lines = []

    finished = False

    for i in range(99):
        path = data_dir + "real_data.txt"
        with open(path, 'r') as f:
            for line in f:
                # line = line[:1]
                if tokenize:
                    line = tokenize_string_(line)
                else:
                    line = tuple(line)

                if len(line) > max_length:
                    line = line[:max_length]

                lines.append(line + ( ('`',)*(max_length-len(line)) ) )

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break

    np.random.shuffle(lines)

    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char, count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    for i in range(100):
        print(filtered_lines[i])

    return filtered_lines, charmap, inv_charmap

def load_dataset_from_file(max_length, max_n_examples, tokenize, max_vocab_size, data_dir):
    print("loading dataset...")

    lines = []

    finished = False

    for i in range(99):
        path = data_dir
        with open(path, 'r') as f:
            for line in f:
                # line = line[:1]
                if tokenize:
                    line = tokenize_string_(line)
                else:
                    line = one_by_one_string_(line)

                if len(line) > max_length:
                    line = line[:max_length]

                lines.append(line + ( ('`',)*(max_length-len(line)) ) )

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break

    np.random.shuffle(lines)

    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char, count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    for i in range(100):
#        print(filtered_lines[i])
        pass

    return filtered_lines, charmap, inv_charmap

def load_dataset_realtrain(max_length, max_n_examples, tokenize, max_vocab_size, data_dir):
    print("loading dataset...")

    lines = []

    finished = False

    for i in range(99):
        path = data_dir + "real_data_train.txt"
        with open(path, 'r') as f:
            for line in f:
                # line = line[:1]
                if tokenize:
                    line = tokenize_string_(line)
                else:
                    line = tuple(line)

                if len(line) > max_length:
                    line = line[:max_length]

                lines.append(line + ( ('`',)*(max_length-len(line)) ) )

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break

    np.random.shuffle(lines)

    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char, count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

#    for i in range(100):
#        print(filtered_lines[i])

    return filtered_lines, charmap, inv_charmap



def load_dataset_realtrain_partdata(max_length, max_n_examples, tokenize, max_vocab_size, data_dir,part_num):
    print("loading dataset...")

    lines = []

    finished = False

    for i in range(99):
        path = data_dir + "real_data_train.txt"
        with open(path, 'r') as f:
            for line in f:
                # line = line[:1]
                if tokenize:
                    line = tokenize_string_(line)
                else:
                    line = tuple(line)

                if len(line) > max_length:
                    line = line[:max_length]

                lines.append(line + ( ('`',)*(max_length-len(line)) ) )

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break

    np.random.shuffle(lines)

    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char, count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))
    filtered_lines = sample(filtered_lines,part_num)
    for i in range(100):
        print(filtered_lines[i])

    return filtered_lines, charmap, inv_charmap