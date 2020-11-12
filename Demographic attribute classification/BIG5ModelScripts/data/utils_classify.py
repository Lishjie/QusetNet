# -*- coding: utf-8 -*-

import time
import Config
import numpy as np
import sys
import utils_classify as utils
from sklearn.preprocessing import OneHotEncoder
sys.path.append('../')

config = Config.Big5Config()############

answers = np.load(config.path+'compliance_data_gai.npy')
features = np.load(config.path+'compliance_feature_gai.npy')
print("shape answers and features:",np.shape(answers),np.shape(features))

class DataStream(object):
    def __init__(self,answers=answers,features=features,config=config):
        # check answers and features has same dim
        assert len(answers) == len(features), "error dim"
        self.answers = answers
        self.features = features
        self.batch_size = config.batch_size
        self.answers_dim = config.answer_dim
        self.feature_dim = config.feature_dim
        self.max_dim = 40*self.answers_dim     # max expend dim
        self.auxiliary_feature = None
        # calculate the expend dim of every feature
        self.expend_dim = [int(self.max_dim // dim) for dim in self.feature_dim]
        self.features_onehot = []  

    def get_user_answers(self, user_id):
        return [answer - 1 for answer in self.answers[user_id]]

    def get_user_features(self, user_id):
        return [feature - 1 for feature in self.features[user_id]]

    def features_encoder(self):
        encoder = OneHotEncoder()
        for col_index in range(self.features.shape[1]):
            self.features_onehot.append(encoder.fit_transform(np.reshape(
                self.features[:, col_index], (-1, 1)
            )).toarray())

    def get_user_feature_expend(self, user_id, pop=False):
        features = self.get_user_features(user_id)
        # print(self.feature_dim)
        if pop:
            features.pop()
            # print(np.array(features).shape)
        result = []

        for i, val_feature in enumerate(features):
          

            temp1 = np.eye(self.feature_dim[i])[val_feature]
            temp2 = np.tile(np.expand_dims(temp1, axis=1), self.expend_dim[i])
            temp3 = np.resize(temp2, -1)
            if len(temp3) < self.max_dim:
                temp3 = np.concatenate((temp3, np.zeros(self.max_dim - len(temp3))), axis=0)
            result.append(temp3)
        return np.array(result)

    def prepare_auxiliary_feature(self, feature_index):
        encoder = OneHotEncoder()
        answers = utils.answers - 1
        features = utils.features - 1
        label = encoder.fit_transform(np.reshape(
            features[:, feature_index], (-1, 1))).toarray()
        label_ = features[:, feature_index]
        features = np.delete(features, feature_index, axis=1)
        features = encoder.fit_transform(features).toarray()
        self.auxiliary_feature = np.concatenate((answers, features), axis=1)


    # feed method:
    def mask_feed_dic(self, user_id, arr, feature_index):
        if len(self.features_onehot) == 0:
            self.features_encoder()
        batch_input_answer = []      # [0,1,2,3,...47]
        batch_input_answer_re = []   # [47,46,45,44,...1]

        batch_input_feature = []     # [48,49,50,...71]
        batch_input_feature_re = []  # [72,71,70,69,...48]

        batch_label_feature = []

        for i in range(self.batch_size):
            batch_label_feature.append(
                self.features[arr[user_id]][feature_index])

            batch_input_answer.append([answer - 1 for answer in self.answers[arr[user_id]]][:])
            batch_input_answer_re.append([answer - 1 for answer in self.answers[arr[user_id]][1:]].copy())
            batch_input_answer_re[i].reverse()

            batch_input_feature.append([feature for feature in self.get_user_feature_expend(arr[user_id], True)])
            batch_input_feature_re.append([feature for feature in self.get_user_feature_expend(arr[user_id], False)])
            batch_input_feature_re[i] = np.flip(batch_input_feature_re[i], 0)

            user_id += 1

        return np.array(batch_input_answer), np.array(batch_input_answer_re), \
            np.array(batch_input_feature), np.array(batch_input_feature_re), \
            np.array(batch_label_feature)

    def mask_feed_dic_with_auxiliary_features(self, user_id, arr, feature_index):
        batch_input_answer = []      # [0,1,2,3,...47]
        batch_input_answer_re = []   # [47,46,45,44,...1]

        batch_input_feature = []     # [48,49,50,...71]
        batch_input_feature_re = []  # [72,71,70,69,...48]

        batch_label_feature = []
        batch_auxiliary_features = []

        for i in range(self.batch_size):
            batch_label_feature.append(self.features[arr[user_id]][feature_index])

            batch_input_answer.append([answer - 1 for answer in self.answers[arr[user_id]]][:])
            batch_input_answer_re.append([answer - 1 for answer in self.answers[arr[user_id]][1:]].copy())
            batch_input_answer_re[i].reverse()

            batch_input_feature.append([feature for feature in self.get_user_feature_expend(arr[user_id], True)])
            batch_input_feature_re.append([feature for feature in self.get_user_feature_expend(arr[user_id], False)])
            batch_input_feature_re[i] = np.flip(batch_input_feature_re[i], 0)

            batch_auxiliary_features.append(self.auxiliary_feature[arr[user_id],:])

            user_id += 1

        return np.array(batch_input_answer), np.array(batch_input_answer_re), \
            np.array(batch_input_feature), np.array(batch_input_feature_re), \
            np.array(batch_label_feature), np.array(batch_auxiliary_features)

    def tran_feed_dic(self, start, end, feature_index):
        batch_input_answer = []
        batch_label_feature = []

        for i in range(start, end):
            batch_input_answer.append([answer - 1 for answer in self.answers[i]][:]+ [feature - 1 for feature in self.features[i]][:])
            batch_label_feature.append(self.features[i][feature_index])

        batch_input_answer = np.delete(batch_input_answer, self.answers.shape[1] + feature_index, axis=1)

        return np.array(batch_input_answer), np.array(batch_label_feature)

    # feed method
    def oppo_feed_dic(self, user_id, arr):
        batch_input_answer = []      # [0,1,2,3,...47]
        batch_label_answer = []      # [1,2,3,4,...47]
        batch_input_answer_re = []   # [47,46,45,44,...1]
        batch_label_answer_re = []   # [47,46,45,44,...0]

        batch_input_feature = []     # [48,49,50,...71]
        batch_label_feature = []     # [49,50,51,...72]
        batch_input_feature_re = []  # [72,71,70,69,...48]
        batch_label_feature_re = []  # [71,70,69,68,...48]

        for i in range(self.batch_size):
            batch_input_answer.append([answer - 1 for answer in self.answers[arr[user_id]]][:])
            batch_label_answer.append([answer - 1 for answer in self.answers[arr[user_id]]][1:])
            batch_input_answer_re.append([answer - 1 for answer in self.answers[arr[user_id]][1:]].copy())
            batch_input_answer_re[i].reverse()
            batch_label_answer_re.append(batch_input_answer[i].copy())
            batch_label_answer_re[i].reverse()

            batch_input_feature.append([feature for feature in self.get_user_feature_expend(arr[user_id], True)])
            batch_label_feature.append([feature - 1 for feature in self.features[arr[user_id]]][:])
            batch_input_feature_re.append([feature for feature in self.get_user_feature_expend(arr[user_id], False)])
            batch_input_feature_re[i] = np.flip(batch_input_feature_re[i], 0)
            batch_label_feature_re.append([feature - 1 for feature in self.features[arr[user_id]]])
            batch_label_feature_re[i].pop()
            batch_label_feature_re[i].reverse()

            user_id += 1
        return np.array(batch_input_answer), np.array(batch_label_answer), \
            np.array(batch_input_feature), np.array(batch_label_feature), \
            np.array(batch_input_answer_re), np.array(batch_input_feature_re), \
            np.array(batch_label_answer_re), np.array(batch_label_feature_re)

    # feed method

    def feed_dic(self, user_id):
        batch_input_answer = []      # [0,1,2,3,...47]
        batch_label_answer = []      # [1,2,3,4,...47]
        batch_input_answer_re = []   # [47,46,45,44,...1]
        batch_label_answer_re = []   # [47,46,45,44,...0]

        batch_input_feature = []     # [48,49,50,...71]
        batch_label_feature = []     # [49,50,51,...72]
        batch_input_feature_re = []  # [72,71,70,69,...48]
        batch_label_feature_re = []  # [71,70,69,68,...48]

        for i in range(self.batch_size):
            batch_input_answer.append([answer - 1 for answer in self.answers[user_id]][:])
            batch_label_answer.append([answer - 1 for answer in self.answers[user_id]][1:])
            batch_input_answer_re.append([answer - 1 for answer in self.answers[user_id][1:]].copy())
            batch_input_answer_re[i].reverse()
            batch_label_answer_re.append(batch_input_answer[i].copy())
            batch_label_answer_re[i].reverse()

            batch_input_feature.append([feature for feature in self.get_user_feature_expend(user_id, True)])
            batch_label_feature.append([feature - 1 for feature in self.features[user_id]][:])
            batch_input_feature_re.append([feature for feature in self.get_user_feature_expend(arr[user_id], False)])
            batch_input_feature_re[i] = np.flip(batch_input_feature_re[i], 0)
            batch_label_feature_re.append([feature - 1 for feature in self.features[user_id]])
            batch_label_feature_re[i].pop()
            batch_label_feature_re[i].reverse()

            user_id += 1
        return np.array(batch_input_answer), np.array(batch_label_answer), \
            np.array(batch_input_feature), np.array(batch_label_feature), \
            np.array(batch_input_answer_re), np.array(batch_input_feature_re), \
            np.array(batch_label_answer_re), np.array(batch_label_feature_re)
"""
if __name__ == "__main__":
    arr = np.arange(answers.shape[0])
    data = DataStream()
    data.prepare_auxiliary_feature(0)

    a, b, c, d, e, f = data.mask_feed_dic_with_auxiliary_features(0, np.arange(7187), 3)
"""
