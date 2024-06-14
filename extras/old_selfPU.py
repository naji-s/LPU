# import logging
# import os
# import sys
# import tqdm


# import sklearn.metrics
# import torch.utils.data
# import unittest.mock

# import LPU.external_libs.Self_PU.functions
# import LPU.external_libs.Self_PU.mean_teacher
# import LPU.external_libs.Self_PU.models
# import LPU.external_libs.Self_PU.utils



# import numpy as np
# import pandas as pd
# import sklearn.model_selection
# import torch
# import LPU.constants
# import LPU.external_libs
# import LPU.models.geometric.elkanGGPC
# import LPU.models.lpu_model_base
# import LPU.external_libs.Self_PU.datasets
# import LPU.external_libs.Self_PU.train
# import LPU.external_libs.Self_PU.mean_teacher.losses
# import LPU.external_libs.Self_PU.mean_teacher.ramps

# torch.set_default_dtype(LPU.constants.DTYPE)

# LOG = logging.getLogger(__name__)

# EPSILON = 1e-16

# class MNIST_Dataset(torch.utils.data.Dataset):

#     def __init__(self, n_labeled, n_unlabeled, trainX, trainY, testX, testY, type="noisy", split="train", mode="N", ids=None, pn=False, increasing=False, replacement=True, top = 0.5, flex = 0, pickout=True, seed = None):

#         self.X, self.Y, self.T, self.oids, self.prior = make_dataset(((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
#         assert np.all(self.oids == np.linspace(0, len(self.X) - 1, len(self.X)))
#         self.clean_ids = []
#         # self.Y_origin = self.Y
#         self.P = self.Y.copy()
#         self.type = type
#         if (ids is None):
#             self.ids = self.oids
#         else:
#             self.ids = np.array(ids)

#         self.split = split
#         self.mode = mode
#         self.pos_ids = self.oids[self.Y == 1]

#         self.pid = self.pos_ids
#         if len(self.ids) != 0:
#             self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
#         else:
#             self.uid = []
#         print(len(self.uid))
#         print(len(self.pid))
#         self.sample_ratio = len(self.uid) // len(self.pid)  + 1
#         print(self.sample_ratio)
#         print("origin:", len(self.pos_ids), len(self.ids))
#         self.increasing = increasing
#         self.replacement = replacement
#         self.top = top
#         self.flex = flex
#         self.pickout = pickout

#         self.pick_accuracy = []
#         self.result = -np.ones(len(self))

#     def copy(self, dataset):
#         ''' Copy random sequence
#         '''
#         self.X, self.Y, self.T, self.oids = dataset.X.copy(), dataset.Y.copy(), dataset.T.copy(), dataset.oids.copy()
#         self.P = self.Y.copy()
        
#     def __len__(self):
#         if self.type == 'noisy':

#             return len(self.pid) * self.sample_ratio
#         else:
#             return len(self.ids)

#     def set_type(self, type):
#         self.type = type

#     def update_prob(self, result):
#         rank = np.empty_like(result)
#         rank[np.argsort(result)] = np.linspace(0, 1, len(result))
#         #print(rank)
#         if (len(self.pos_ids) > 0):
#             rank[self.pos_ids] = -1
#         self.result = rank
        
#     def shuffle(self):
#         perm = np.random.permutation(len(self.uid))
#         self.uid = self.uid[perm]

#         perm = np.random.permutation(len(self.pid))
#         self.pid = self.pid[perm]

#     def __getitem__(self, idx): 
#         #print(idx)
#         # self.ids[idx]是真实的行索引
#         # 始终使用真实的行索引去获得数据

#         # 1901 保持比例
#         if self.type == 'noisy':
#             if (idx % self.sample_ratio == 0):
#                 try:
#                     index = self.pid[idx // self.sample_ratio]
#                     id = self.ids[idx // self.sample_ratio]
#                 except IndexError:
#                     print(idx)
#                     print(self.sample_ratio)
#                     print(len(self.pid))
#             else:
#                 index = self.uid[idx - (idx // self.sample_ratio + 1)]
#                 id = self.ids[idx - (idx // self.sample_ratio + 1)]
#             return self.X[index], self.Y[index], self.P[index], self.T[index], id, self.result[index]
#         else:
#             return self.X[self.ids[idx]], self.Y[self.ids[idx]], self.P[self.ids[idx]], self.T[self.ids[idx]], self.ids[idx], self.result[self.ids[idx]]


#     def reset_ids(self):
#         ''' Using all origin ids
#         '''
#         self.ids = self.oids.copy()

#     def set_ids(self, ids):
#         ''' Set specific ids
#         '''
#         self.ids = np.array(ids).copy()
#         if len(ids) > 0:
#             self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
#             self.pid = np.intersect1d(self.ids[self.Y[self.ids] == 1], self.ids)
#             if len(self.pid) == 0:
#                 self.sample_ratio = 10000000000
#             else:
#                 self.sample_ratio = int(len(self.uid) / len(self.pid)) + 1
#     def reset_labels(self):
#         ''' Reset Y labels
#         '''
#         self.P = self.Y.copy()

#     def update_ids(self, results, epoch, ratio=None, lt = 0, ht = 0):
#         if not self.replacement or self.increasing:
#             percent = min(epoch / 100, 1) # 决定抽取数据的比例
#         else:
#             percent = 1
#         if ratio == None:
#             ratio = self.prior

#         if self.mode == 'N':
#             self.reset_labels()
#             n_neg = int((len(self.oids) - len(self.pos_ids)) * (1 - ratio) * percent) # 决定抽取的数量
#             if self.replacement:
#                 # 如果替换的话，抽取n_neg个
#                 neg_ids = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)[:n_neg]
#             else:
#                 # 否则抽取n_neg - #ids
#                 neg_ids = np.setdiff1d(np.argsort(results), self.ids, assume_unique=True)[:n_neg]
#             # 变成向量
#             neg_ids = np.array(neg_ids) 
#             neg_label = self.T[neg_ids] # 获得neg_ids的真实标签
#             correct = np.sum(neg_label < 1) # 抽取N的时候真实标签为-1
#             print("Correct: {}/{}".format(correct, len(neg_ids))) # 打印
#             if self.replacement:
#                 self.ids = np.concatenate([self.pos_ids[:len(self.pos_ids) // 2], neg_ids]) # 如果置换的话，在ids的基础上加上neg_ids
#             else:
#                 if len(self.ids) == 0: self.ids = np.concatenate([self.ids, self.pos_ids[:len(self.pos_ids) // 2]]) # 如果为空的话则首先加上pos_ids
#                 self.ids = np.concatenate([self.ids, neg_ids])

#             self.ids = self.ids.astype(int) # 为了做差集
#             out = np.setdiff1d(self.oids, self.ids) # 计算剩下的ids的数量并返回
#             #assert len(np.intersect1d(self.ids, out)) == 0 # 要求两者不能有重合
#             return out

#         elif self.mode == 'P':
#             self.reset_labels()
#             n_pos = int((len(self.oids) - len(self.pos_ids)) * (1 - ratio) * percent) # 决定抽取的数量

#             if self.replacement:
#                 # 如果替换的话，抽取n_neg个
#                 pos_ids = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)[-n_pos:]
#             else:
#                 # 否则抽取n_neg - #ids
#                 pos_ids = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)[-(n_pos - len(self.ids)):]

#             # 变成向量
#             pos_ids = np.array(pos_ids) 
#             pos_label = self.T[pos_ids] # 获得neg_ids的真实标签
#             correct = np.sum(pos_label == 1) # 抽取N的时候真实标签为-1

#             self.Y[pos_ids] = 1 # 将他们标注为1
#             print("Correct: {}/{}".format(correct, len(pos_ids))) # 打印
#             if self.replacement:
#                 self.ids = np.concatenate([self.pos_ids[:len(self.pos_ids) // 2], pos_ids]) # 如果置换的话，在ids的基础上加上neg_ids
#             else:
#                 if len(self.ids) == 0: self.ids = np.concatenate([self.ids, self.pos_ids[:len(self.pos_ids) // 2]]) # 如果为空的话则首先加上pos_ids
#                 self.ids = np.concatenate([self.ids, pos_ids])
            
#             self.ids = self.ids.astype(int) # 为了做差集
#             out = np.setdiff1d(self.oids, self.ids) # 计算剩下的ids的数量并返回
#             #assert len(np.intersect1d(self.ids, out)) == 0 # 要求两者不能有重合
#             return out

#         elif self.mode == 'A':
#             self.reset_labels()
#             n_all = int((len(self.oids) - len(self.pos_ids)) * (1 - ratio) * percent * self.top) # 决定抽取的数量
#             confident_num = int(n_all * (1 - self.flex))
#             noisy_num = int(n_all * self.flex)
#             if self.replacement:
#                 # 如果替换的话，抽取n_pos个
#                 #print(np.argsort(results))
#                 #print(np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True))
#                 al = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)
#                 neg_ids = al[:confident_num]
#                 pos_ids = al[-confident_num:]
#             else:
#                 # 否则抽取n_pos - #ids
#                 al = np.setdiff1d(np.argsort(results), self.ids, assume_unique=True)
#                 neg_ids = al[:(confident_num - len(self.ids) // 2)]
#                 pos_ids = al[-(confident_num - len(self.ids) // 2):]

#             # 变成向量
#             pos_ids = np.array(pos_ids) 
#             pos_label = self.T[pos_ids] # 获得neg_ids的真实标签
#             pcorrect = np.sum(pos_label == 1) # 抽取N的时候真实标签为-1

#             neg_ids = np.array(neg_ids) 
#             neg_label = self.T[neg_ids] # 获得neg_ids的真实标签
#             ncorrect = np.sum(neg_label < 1)

#             self.P[pos_ids] = 1 # 将他们标注为1
#             print("P Correct: {}/{}".format(pcorrect, len(pos_ids))) # 打印
#             print("N Correct: {}/{}".format(ncorrect, len(neg_ids)))

#             self.pick_accuracy.append((pcorrect + ncorrect) * 1.0 / (len(pos_ids) * 2))
#             if self.replacement:
#                 #self.ids = np.concatenate([self.pos_ids, pos_ids, neg_ids]) # 如果置换的话，在ids的基础上加上neg_ids
#                 self.ids = np.concatenate([pos_ids, neg_ids]) 
#             else:
#                 #if len(self.ids) == 0: self.ids = np.concatenate([self.ids, self.pos_ids]) # 如果为空的话则首先加上pos_ids
#                 #self.ids = np.concatenate([self.ids, pos_ids, neg_ids])
#                 self.ids = np.concatenate([self.ids, pos_ids, neg_ids])
            
#             self.ids = self.ids.astype(int) # 为了做差集
#             if self.pickout:
#                 out = np.setdiff1d(self.oids, self.ids) # 计算剩下的ids的数量并返回
#             else:
#                 out = self.oids
#             if noisy_num > 0:
#                 noisy_select = out[np.random.permutation(len(out))][:noisy_num]
#                 self.P[np.intersect1d(results >= 0.5, noisy_select)] = 1
#                 self.ids = np.concatenate([self.ids, noisy_select], 0)
#                 if self.pickout:
#                     out = np.setdiff1d(self.oids, self.ids)
#             if self.pickout:
#                 assert len(np.intersect1d(self.ids, out)) == 0 # 要求两者不能有重合
#             return out


#         elif self.mode == 'T':
#             self.reset_labels()
            
#             al = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)
#             print(lt)
#             print(ht)
#             negative_confident_num = int(lt * len(al))
#             positive_confident_num = int((1-ht) * len(al))
#             neg_ids = al[:negative_confident_num]
#             pos_ids = al[len(al)-positive_confident_num:]

#             # 变成向量
#             pos_ids = np.array(pos_ids) 
#             pos_label = self.T[pos_ids] # 获得neg_ids的真实标签
#             pcorrect = np.sum(pos_label == 1) # 抽取N的时候真实标签为-1

#             neg_ids = np.array(neg_ids) 
#             neg_label = self.T[neg_ids] # 获得neg_ids的真实标签
#             ncorrect = np.sum(neg_label < 1)

#             self.P[pos_ids] = 1 # 将他们标注为1
#             print("P Correct: {}/{}".format(pcorrect, len(pos_ids))) # 打印
#             print("N Correct: {}/{}".format(ncorrect, len(neg_ids)))

#             self.pick_accuracy.append((pcorrect + ncorrect) * 1.0 / (len(pos_ids) * 2))
#             if self.replacement:
#                 #self.ids = np.concatenate([self.pos_ids, pos_ids, neg_ids]) # 如果置换的话，在ids的基础上加上neg_ids
#                 self.ids = np.concatenate([pos_ids, neg_ids]) 
#             else:
#                 #if len(self.ids) == 0: self.ids = np.concatenate([self.ids, self.pos_ids]) # 如果为空的话则首先加上pos_ids
#                 #self.ids = np.concatenate([self.ids, pos_ids, neg_ids])
#                 self.ids = np.concatenate([self.ids, pos_ids, neg_ids])
            
#             self.ids = self.ids.astype(int) # 为了做差集
#             if self.pickout:
#                 out = np.setdiff1d(self.oids, self.ids) # 计算剩下的ids的数量并返回
#             else:
#                 out = self.oids
#             if self.pickout:
#                 assert len(np.intersect1d(self.ids, out)) == 0 # 要求两者不能有重合
#             return out
        
# # class MNIST_Dataset_FixSample(torch.utils.data.Dataset):

# #     def __init__(self, n_labeled, n_unlabeled, trainX, trainY, testX, testY, type="noisy", split="train", mode="A", ids=None, pn=False, increasing=False, replacement=True, top = 0.5, flex = 0, pickout=True, seed = None):

# #         self.X, self.Y, self.T, self.oids, self.prior = make_dataset(((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
# #         assert np.all(self.oids == np.linspace(0, len(self.X) - 1, len(self.X)))
# #         self.clean_ids = []
# #         #self.Y_origin = self.Y
# #         self.P = self.Y.copy()
# #         self.type = type
# #         if (ids is None):
# #             self.ids = self.oids
# #         else:
# #             self.ids = np.array(ids)

# #         self.split = split
# #         self.mode = mode
# #         self.pos_ids = self.oids[self.Y == 1]


# #         self.pid = self.pos_ids
# #         if len(self.ids) != 0:
# #             self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
# #         else:
# #             self.uid = []
# #         print(len(self.uid))
# #         print(len(self.pid))
# #         self.sample_ratio = len(self.uid) // len(self.pid)  + 1
# #         print(self.sample_ratio)
# #         print("origin:", len(self.pos_ids), len(self.ids))
# #         self.increasing = increasing
# #         self.replacement = replacement
# #         self.top = top
# #         self.flex = flex
# #         self.pickout = pickout

# #         self.pick_accuracy = []
# #         self.result = -np.ones(len(self))

# #         self.random_count = 0
# #     def copy(self, dataset):
# #         ''' Copy random sequence
# #         '''
# #         self.X, self.Y, self.T, self.oids = dataset.X.copy(), dataset.Y.copy(), dataset.T.copy(), dataset.oids.copy()
# #         self.P = self.Y.copy()
# #     def __len__(self):
# #         if self.type == 'noisy':

# #             #return len(self.uid) * 2
# #             return len(self.pid) * self.sample_ratio
# #         else:
# #             return len(self.ids)

# #     def set_type(self, type):
# #         self.type = type

# #     def update_prob(self, result):
# #         rank = np.empty_like(result)
# #         rank[np.argsort(result)] = np.linspace(0, 1, len(result))
# #         #print(rank)
# #         if (len(self.pos_ids) > 0):
# #             rank[self.pos_ids] = -1
# #         self.result = rank
        
# #     def shuffle(self):
# #         perm = np.random.permutation(len(self.uid))
# #         self.uid = self.uid[perm]

# #         perm = np.random.permutation(len(self.pid))
# #         self.pid = self.pid[perm]

# #     def __getitem__(self, idx): 
# #         #print(idx)
# #         # self.ids[idx]是真实的行索引
# #         # 始终使用真实的行索引去获得数据

# #         # 1901 保持比例
# #         if self.type == 'noisy':
# #             '''
# #             if (idx % 2 == 0):
# #                 index = self.pid[idx % 1000]
# #             else:
# #                 index = self.uid[idx - (idx // 2 + 1)]

# #             '''
# #             if (idx % self.sample_ratio == 0):
# #                 index = self.pid[idx // self.sample_ratio]
# #                 id = 0
# #             else:
# #                 index = self.uid[idx - (idx // self.sample_ratio + 1)]
                
# #             return self.X[index], self.Y[index], self.P[index], self.T[index], index, 0
# #         else:
# #             return self.X[self.ids[idx]], self.Y[self.ids[idx]], self.P[self.ids[idx]], self.T[self.ids[idx]], self.ids[idx], 0


# #     def reset_ids(self):
# #         ''' Using all origin ids
# #         '''
# #         self.ids = self.oids.copy()

# #     def set_ids(self, ids):
# #         ''' Set specific ids
# #         '''
# #         self.ids = np.array(ids).copy()
# #         if len(ids) > 0:
# #             self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
# #             self.pid = np.intersect1d(self.ids[self.Y[self.ids] == 1], self.ids)
# #             if len(self.pid) == 0:
# #                 self.sample_ratio = 10000000000
# #             else:
# #                 self.sample_ratio = int(len(self.uid) / len(self.pid)) + 1
# #     def reset_labels(self):
# #         ''' Reset Y labels
# #         '''
# #         self.P = self.Y.copy()

# #     def update_ids(self, results, epoch, ratio=None, lt = 0, ht = 0):
# #         if not self.replacement or self.increasing:
# #             percent = min(epoch / 100, 1) # 决定抽取数据的比例
# #         else:
# #             percent = 1
# #         if ratio == None:
# #             ratio = self.prior
# #         self.reset_labels()
# #         n_all = int((len(self.oids) - len(self.pos_ids)) * (1 - ratio) * percent * self.top) # 决定抽取的数量
# #         confident_num = int(n_all * (1 - self.flex))
# #         noisy_num = int(n_all * self.flex)
# #         if self.replacement:
# #             # 如果替换的话，抽取n_pos个
# #             #print(np.argsort(results))
# #             #print(np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True))
# #             al = np.setdiff1d(np.argsort(results), self.pos_ids, assume_unique=True)
# #             neg_ids = al[:confident_num]
# #             pos_ids = al[-confident_num:]
# #         else:
# #             # 否则抽取n_pos - #ids
# #             al = np.setdiff1d(np.argsort(results), self.ids, assume_unique=True)
# #             neg_ids = al[:(confident_num - len(self.ids) // 2)]
# #             pos_ids = al[-(confident_num - len(self.ids) // 2):]

# #         # 变成向量
# #         pos_ids = np.array(pos_ids) 
# #         pos_label = self.T[pos_ids] # 获得neg_ids的真实标签
# #         pcorrect = np.sum(pos_label == 1) # 抽取N的时候真实标签为-1

# #         neg_ids = np.array(neg_ids) 
# #         neg_label = self.T[neg_ids] # 获得neg_ids的真实标签
# #         ncorrect = np.sum(neg_label < 1)

# #         self.P[pos_ids] = 1 # 将他们标注为1
# #         print("P Correct: {}/{}".format(pcorrect, len(pos_ids))) # 打印
# #         print("N Correct: {}/{}".format(ncorrect, len(neg_ids)))

# #         self.pick_accuracy.append((pcorrect + ncorrect) * 1.0 / (len(pos_ids) * 2))
# #         if self.replacement:
# #             #self.ids = np.concatenate([self.pos_ids, pos_ids, neg_ids]) # 如果置换的话，在ids的基础上加上neg_ids
# #             self.ids = np.concatenate([pos_ids, neg_ids]) 
# #         else:
# #             #if len(self.ids) == 0: self.ids = np.concatenate([self.ids, self.pos_ids]) # 如果为空的话则首先加上pos_ids
# #             #self.ids = np.concatenate([self.ids, pos_ids, neg_ids])
# #             self.ids = np.concatenate([self.ids, pos_ids, neg_ids])
        
# #         self.ids = self.ids.astype(int) # 为了做差集
# #         if self.pickout:
# #             out = np.setdiff1d(self.oids, self.ids) # 计算剩下的ids的数量并返回
# #         else:
# #             out = self.oids
# #         if noisy_num > 0:
# #             noisy_select = out[np.random.permutation(len(out))][:noisy_num]
# #             self.P[np.intersect1d(results >= 0.5, noisy_select)] = 1
# #             self.ids = np.concatenate([self.ids, noisy_select], 0)
# #             if self.pickout:
# #                 out = np.setdiff1d(self.oids, self.ids)
# #         if self.pickout:
# #             assert len(np.intersect1d(self.ids, out)) == 0 # 要求两者不能有重合
# #         return out

# def make_dataset(dataset, n_labeled, n_unlabeled, mode="train", pn=False, seed = None):
        
#     def make_PU_dataset_from_binary_dataset(x, y, l):
#         # labels = np.unique(y)
#         X, Y, L = np.asarray(x, dtype=LPU.constants.NUMPY_DTYPE), np.asarray(y, dtype=int), np.asarray(l, dtype=int)
#         perm = np.random.permutation(len(X))
#         X, Y, L = X[perm], Y[perm], L[perm]

#         prior = l.mean() / l[y==1].mean() 
#         ids = np.arange(len(X))
#         return X, L, Y, ids, prior


#     def make_PN_dataset_from_binary_dataset(x, y):
#         labels = np.unique(y)
#         positive, negative = labels[1], labels[0]
#         X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
#         if seed is not None:
#             np.random.seed(seed)
#         perm = np.random.permutation(len(X))
#         X, Y = X[perm], Y[perm]
#         n_p = (Y == positive).sum()
#         n_n = (Y == negative).sum()
#         Xp = X[Y == positive][:n_p]
#         Xn = X[Y == negative][:n_n]
#         X = np.asarray(np.concatenate((Xp, Xn)), dtype=np.float32)
#         Y = np.asarray(np.concatenate((np.ones(n_p), -np.ones(n_n))), dtype=np.int32)
#         ids = np.array([i for i in range(len(X))])
#         return X, Y, Y, ids

        

#     (_trainX, _trainY, _trainL), (_testX, _testY, _testL) = dataset
#     prior = None
#     if (mode == 'train'):
#         if not pn:
#             X, Y, T, ids, prior = make_PU_dataset_from_binary_dataset(_trainX, _trainY, _trainL)
#         else:
#             raise NotImplementedError("Not implemented yet.")
#     else:
#         X, Y, T, ids  = make_PN_dataset_from_binary_dataset(_testX, _testY)
#     return X, Y, T, ids, prior



# class selfPUModifiedDataset(MNIST_Dataset):
#     def __init__(self, trainX, trainY, trainL, testX, testY, testL, type="noisy", split="train", mode=None, ids=None, pn=False, increasing=False, replacement=True, top = 0.5, flex = 0, pickout=True, seed = None):
#         # super().__init__(trainX, trainY, testX, testY, type=type, split=split, mode=mode, ids=ids, pn=pn, increasing=increasing, replacement=replacement, top=top, flex=flex, pickout=pickout, seed=seed)
#         n_labeled = trainL.sum().to(int)
#         n_unlabeled = len(trainL) - n_labeled

#         # super().__init__(n_labeled, n_unlabeled, trainX, trainY, testX, testY, type=type, split=split, mode=mode, ids=ids, pn=pn, increasing=increasing, replacement=replacement, top = top, flex = flex, pickout=pickout, seed = seed)
#         # if split == "train":
#         #     self.X, self.Y, self.T, self.oids, self.prior = LPU.external_libs.Self_PU.datasets.make_dataset(((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
#         # elif split == "test":
#         #     self.X, self.Y, self.T, self.oids, self.prior = LPU.external_libs.Self_PU.datasets.make_dataset(((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
#         # else:
#         #     raise ValueError("split should be either 'train' or 'test'")
#         if set(np.unique(trainL)).issubset({0, 1}):
#             trainY = 2 * trainY - 1
#             trainL = 2 * trainL - 1
#             testY = 2 * testY - 1
#             testL = 2 * testL - 1

#         if split == "train":
#             self.X, self.Y, self.T, self.oids, self.prior = make_dataset(((trainX, trainY, trainL), (testX, testY, testL)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
#         elif split == "test":
#             self.X, self.Y, self.T, self.oids, self.prior = make_dataset(((trainX, trainY, trainL), (testX, testY, testL)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
#         else:
#             raise ValueError("split should be either 'train' or 'test'")
#         assert np.all(self.oids == np.linspace(0, len(self.X) - 1, len(self.X)))
#         self.clean_ids = []
#         # self.Y_origin = self.Y
#         self.P = self.Y.copy()
#         self.type = type
#         if (ids is None):
#             self.ids = self.oids
#         else:
#             self.ids = np.array(ids)

#         self.split = split
#         self.mode = mode
#         self.pos_ids = self.oids[self.Y == 1]

#         self.pid = self.pos_ids
#         if len(self.ids) != 0:
#             self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
#         else:
#             self.uid = []
#         print(len(self.uid))
#         print(len(self.pid))
#         self.sample_ratio = len(self.uid) // len(self.pid)  + 1
#         print("SAMPLE RATIO:", self.sample_ratio)
#         print("origin:", len(self.pos_ids), len(self.ids), type)
#         self.increasing = increasing
#         self.replacement = replacement
#         self.top = top
#         self.flex = flex
#         self.pickout = pickout

#         self.pick_accuracy = []
#         self.result = -np.ones(len(self))
#         self.result = -np.ones(len(trainX) + len(testX))


# class selfPUModifiedDataset_FixSample(LPU.external_libs.Self_PU.datasets.MNIST_Dataset_FixSample):
#     def __init__(self, trainX, trainY, trainL, testX, testY, testL, type="noisy", split="train", mode=None, ids=None, pn=False, increasing=False, replacement=True, top=0.5, flex=0, pickout=True, seed=None):
#         n_labeled = trainL.sum().to(int)
#         n_unlabeled = len(trainL) - n_labeled

#         if set(np.unique(trainL)).issubset({0, 1}):
#             trainY = 2 * trainY - 1
#             trainL = 2 * trainL - 1
#             testY = 2 * testY - 1
#             testL = 2 * testL - 1

#         if split == "train":
#             self.X, self.Y, self.T, self.oids, self.prior = make_dataset(((trainX, trainY, trainL), (testX, testY, testL)), n_labeled, n_unlabeled, mode=split, pn=pn, seed=seed)
#         elif split == "test":
#             self.X, self.Y, self.T, self.oids, self.prior = make_dataset(((trainX, trainY, trainL), (testX, testY, testL)), n_labeled, n_unlabeled, mode=split, pn=pn, seed=seed)
#         else:
#             raise ValueError("split should be either 'train' or 'test'")

#         assert np.all(self.oids == np.linspace(0, len(self.X) - 1, len(self.X)))
#         self.clean_ids = []
#         self.P = self.Y.copy()
#         self.type = type
#         if (ids is None):
#             self.ids = self.oids
#         else:
#             self.ids = np.array(ids)

#         self.split = split
#         self.mode = mode
#         self.pos_ids = self.oids[self.Y == 1]

#         self.pid = self.pos_ids
#         if len(self.ids) != 0:
#             self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
#         else:
#             self.uid = []
#         print(len(self.uid))
#         print(len(self.pid))
#         self.sample_ratio = len(self.uid) // len(self.pid) + 1
#         print("SAMPLE RATIO:", self.sample_ratio)
#         print("origin:", len(self.pos_ids), len(self.ids), type)
#         self.increasing = increasing
#         self.replacement = replacement
#         self.top = top
#         self.flex = flex
#         self.pickout = pickout

#         self.pick_accuracy = []
#         self.result = -np.ones(len(self))
#         self.result = -np.ones(len(trainX) + len(testX))

# def sigmoid_rampup(current, rampup_length):
#     """   
#     Functions for ramping hyperparameters up or down
#     Each function takes the current training step or epoch, and the
#     ramp length in the same format, and returns a multiplier between
#     0 and 1.
#     Exponential rampup from https://arxiv.org/abs/1610.02242"""

#     if rampup_length == 0:
#         return 1.0
#     else:
#         current = np.clip(current, 0.0, rampup_length)
#         phase = 1.0 - current / rampup_length
#         return float(np.exp(-5.0 * phase * phase))




# class selfPU(LPU.models.lpu_model_base.LPUModelBase):
#     """
#     """
#     def __init__(self, config, single_epoch_steps, *args, **kwargs):
#         super().__init__(**kwargs)
#         self.config = config
#         self.hidden_dim = config.get('hidden_dim', 32)
#         self.learning_rate = config.get('learning_rate', 1e-4)
#         self.model = None
#         self.epochs = config.get('epochs', 200)
#         self.ema_decay = None
#         self.soft_label = config.get('soft_label', False)
#         self.batch_size = config.get('batch_size', 32)
#         self.num_workers = config.get('num_workers', 4)
#         self.model_dir = config.get('model_dir', 'LPU/scripts/selfPU')
#         self.single_epoch_steps = single_epoch_steps
#         self.step = 0 

#     def check_noisy(self, epoch):
#         if epoch >= self.config['self_paced_start']: #and args.turnoff_noisy:
#             return False
#         else:
#             return True
        
#     def set_C(self, holdout_dataloader):
#         try:
#             assert (holdout_dataloader.batch_size == len(holdout_dataloader.dataset)), "There should be only one batch in the dataloader."
#         except AssertionError as e:
#             LOG.error(f"There should be only one batch in the dataloader, but {holdout_dataloader.batch_size} is smaller than {len(holdout_dataloader.dataset)}.")
#             raise e
#         X, l, y, _ = next(iter(holdout_dataloader))
#         self.C = l[y == 1].mean().detach().cpu().numpy()

#     def check_mean_teacher(self, epoch):
#         if not self.config['mean_teacher']:
#             return False
#         elif epoch < self.config['ema_start']:
#             return False 
#         else:
#             return True
        
#     def get_current_consistency_weight(self, epoch):
#         # Consistency ramp-up from https://arxiv.org/abs/1610.02242
#         return self.config['consistency'] * sigmoid_rampup(epoch, self.config['consistency_rampup'])

#     def update_ema_variables(self):
#         self.config['ema_decay'], self.step
#         alpha = min(1 - 1 / (self.config['ema_decay'] + 1), self.config['ema_decay'])
#         for self.ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
#             self.ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

#     def check_self_paced(self, epoch):
#         if self.config['self_paced_stop'] < 0: self_paced_stop = self.config['num_epochs']
#         else: self_paced_stop = self.config['self_paced_stop']
#         if not self.config['self_paced']:
#             return False
#         elif self.config['self_paced'] and epoch >= self_paced_stop:
#             return False
#         elif self.config['self_paced'] and epoch < self.config['self_paced_start']:
#             return False
#         else: return True

#     def predict_proba(self, X):
#         self.model.eval()
#         return torch.sigmoid(self.model(torch.as_tensor(X, dtype=LPU.constants.DTYPE))).detach().numpy().flatten()

#     def predict_prob_y_given_X(self, X):
#         return self.predict_proba(X) / self.C
    
#     def predict_prob_l_given_y_X(self, X):
#         return self.C 

#     def loss_fn(self, X_batch, l_batch):
#         return torch.tensor(0.0)
    
#     def update_dataset(self, student, teacher, dataset_train_clean, dataset_train_noisy, epoch, ratio=0.5, lt = 0, ht = 1):

#         dataset_train_noisy.reset_ids()
#         dataset_train_noisy.set_type("clean")
#         dataloader_train = torch.utils.data.DataLoader(dataset_train_noisy, batch_size=self.config['batch_size']['train'], num_workers=4, shuffle=False, pin_memory=True)
#         if self.config['dataset'] == 'mnist':
#             results = np.zeros(dataloader_train.dataset.ids.max() + 1) # rid.imageid: p_pos # 存储概率结果
#         elif self.config['dataset'] == 'cifar':
#             results = np.zeros(51000) 
#         else:
#             results = np.zeros(len(dataset_train_noisy))
#         student.eval()
#         teacher.eval()
#         # validation #######################
#         with torch.no_grad():
#             for i_batch, (X, _, _, T, ids, _) in enumerate(tqdm.tqdm(dataloader_train)):
#                 #images, lefts, rights, ages, genders, edus, apoes, labels, pu_labels, ids = Variable(sample_batched['mri']).cuda(), Variable(sample_batched['left']).cuda(), Variable(sample_batched['right']).cuda(), Variable(sample_batched['age']).cuda(), Variable(sample_batched['gender']).cuda(), Variable(sample_batched['edu']).cuda(), Variable(sample_batched['apoe']).cuda(), Variable(sample_batched['label']).view(-1).type(torch.LongTensor).cuda(), Variable(sample_batched['pu_label']).view(-1).type(torch.LongTensor).cuda(), sample_batched['id']
#                 X = X.to(self.config['gpu'])
#                 if self.config['dataset'] == 'mnist':
#                     X = X.reshape(X.shape[0], 1, -1)
#                 # ===================forward====================
#                 outputs_s = student(X)
#                 outputs_t = teacher(X)
#                 prob_n_s = torch.sigmoid(outputs_s).view(-1).cpu().numpy()       
#                 prob_n_t = torch.sigmoid(outputs_t).view(-1).cpu().numpy()
#                 #print(np.sum(prob_n_t < 0.5))

#                 if self.check_mean_teacher(epoch):
#                     results[ids.view(-1).numpy()] = prob_n_t
#                 else:
#                     try:
#                         results[ids.view(-1).numpy()] = prob_n_s
#                     except:
#                         breakpoint()
#         # adni_dataset_train.update_labels(results, ratio)
#         # dataset_origin = dataset_train
#         ids_noisy = dataset_train_clean.update_ids(results, epoch, ratio = ratio, ht = ht, lt = lt) # 返回的是noisy ids
#         dataset_train_noisy.set_ids(ids_noisy) # 将noisy ids更新进去
#         dataset_train_noisy.set_type("noisy")
#         dataset_train_noisy.update_prob(results)
#         dataset_train_clean.update_prob(results)
#         assert np.all(dataset_train_noisy.ids == ids_noisy) # 确定更新了
#         #dataloader_origin = DataLoader(dataset_origin, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
#         dataloader_train_clean = torch.utils.data.DataLoader(dataset_train_clean, batch_size=self.config['batch_size']['train'], num_workers=self.config['num_workers'], shuffle=True, pin_memory=True)
#         dataloader_train_noisy = torch.utils.data.DataLoader(dataset_train_noisy, batch_size=self.config['batch_size']['train'], num_workers=self.config['num_workers'], shuffle=False, pin_memory=True)

#         return dataloader_train_clean, dataloader_train_noisy


#     def train(self, clean_loader, noisy_loader, model, ema_model, criterion, consistency_criterion, optimizer, scheduler,  epoch, warmup = False, self_paced_pick = 0):
    
#         self.model.train()
#         self.ema_model.train()
#         consistency_weight = self.get_current_consistency_weight(epoch - 30)
#         print("Learning rate is {}".format(optimizer.param_groups[0]['lr']))
#         if clean_loader:
#             for i, (X, _, Y, T, ids, _) in enumerate(clean_loader):
#                 # measure data loading time
#                 if self.config['gpu'] == None:
#                     X = X#.cuda()
#                     Y = Y#.cuda().float()
#                     T = T#.cuda().long()
#                 else:
#                     X = X.cuda()
#                     Y = Y.cuda().float()
#                     T = T.cuda().long()
#                 if self.config['dataset'] == 'mnist':
#                     X = X.view(X.shape[0], 1, -1)
                

#                 # compute output
#                 output = model(X)
#                 with torch.no_grad():
#                     ema_output = ema_model(X)

#                 consistency_loss = consistency_weight * \
#                 consistency_criterion(output, ema_output) / X.shape[0]
#                 #if epoch >= args.self_paced_start: criterion.update_p(0.5)
#                 _, loss = criterion(output, Y) # 计算loss，使用PU标签
                
#                 #print(output)
#                 # measure accuracy and record loss
                
#                 if self.check_mean_teacher(epoch):
#                     predictions = torch.sign(ema_output).long() # 使用teacher的结果作为预测
#                 else:
#                     predictions = torch.sign(output).long() # 否则使用自己的结果

#                 smx = torch.sigmoid(output) # 计算sigmoid概率
#                 #print(smx)
#                 smx = torch.cat([1 - smx, smx], dim=1) # 组合成预测变量
#                 smxY = ((Y + 1) / 2).long() # 分类结果，0-1分类
#                 if self.config['soft_label']:
#                     aux = - torch.sum(smx * torch.log(smx + 1e-10)) / smx.shape[0]
#                 else:
#                     smxY = smxY.float()
#                     smxY = smxY.view(-1, 1)
#                     smxY = torch.cat([1 - smxY, smxY], dim = 1)
#                     aux = - torch.sum(smxY * torch.log(smx + 1e-10)) / smxY.shape[0] # 计算Xent loss
                    
#                 loss = aux
#                 if self.check_mean_teacher(epoch):
#                     loss += consistency_loss
                
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#                 if self.check_mean_teacher(epoch) and ((i + 1) % int(self.single_epoch_steps / 2 - 1)) == 0:
#                     self.update_ema_variables() # 更新ema参数
#                     self.step += 1

#         all_ls = []
#         all_ys = []
#         all_y_outputs = []
#         try:
#             for i, (X, Y, _, T, ids, p) in enumerate(noisy_loader):
#                 if self.config['gpu'] == None:
#                     X = X#.cuda()
#                     Y = Y#.cuda().float()
#                     T = T#.cuda().long()
#                     p = p#.cuda().float()
#                 else:
#                     X = X.cuda()
#                     Y = Y.cuda().float()
#                     T = T.cuda().long()
#                     p = p.cuda().float()
#                 if self.config['dataset'] == 'mnist':
#                     X = X.view(X.shape[0], 1, -1)
#                 # breakpoint()
#                 output = model(X)

#                 with torch.no_grad():
#                     ema_output = ema_model(X)

#                 consistency_loss = consistency_weight * \
#                 consistency_criterion(output, ema_output) / X.shape[0]
                
#                 #if epoch >= args.self_paced_start: criterion.update_p(0.5)
#                 _, loss = criterion(output, Y)
#                 if self.check_mean_teacher(epoch) and not warmup:
#                     loss += consistency_loss
#                     predictions = torch.sign(ema_output).long()
#                     output = ema_output
#                 else:
#                     predictions = torch.sign(output).long()

#                 #if epoch >= args.self_paced_start

#                 if self.check_noisy(epoch):
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#                 if self.check_mean_teacher(epoch) and ((i + 1) % int(self.single_epoch_steps / 2 - 1)) == 0 and not warmup:
#                     self.update_ema_variables()
#                     self.step += 1
#                 all_ls.append(Y.cpu().numpy())
#                 all_ys.append(T.cpu().numpy())
#                 all_y_outputs.append(torch.sigmoid(output).detach().cpu().numpy())
#             if not warmup: scheduler.step()
#             all_ys = np.concatenate(all_ys)
#             all_ls = np.concatenate(all_ls)
#             all_y_outputs = np.concatenate(all_y_outputs)
#             all_l_outputs = all_y_outputs * self.C
#             all_l_ests = all_y_outputs > 0.5 * self.C
#             all_y_ests = all_y_outputs > 0.5
#             if not warmup: scheduler.step()

#         except Exception as e:
#             print ("WOAH", e)
#             breakpoint()
#             raise e
#         # breakpoint()        
#         # if set(np.unique(all_ls)).issubset({-1, 1}):
#         all_ys = (all_ys + 1) // 2
#         all_ls = (all_ls + 1) // 2
#         # elif not(set(np.unique(all_ls)).issubset({0, 1})):
#         #     raise ValueError(f"Labels should be either in {{-1, 1}} or {{0, 1}}, but {set(np.unique(all_ls))} is given.")

#         all_scores = self._calculate_validation_metrics(y_probs=all_y_outputs, y_vals=all_ys, l_probs=all_l_outputs, l_vals=all_ls, l_ests=all_l_ests, y_ests=all_y_ests)
#         all_scores['overall_loss'] = loss.detach().cpu().numpy() / (i + 1)
#         return all_scores


#     def validate(self, val_loader, model, ema_model, criterion, consistency_criterion, epoch):
#         consistency_weight = self.get_current_consistency_weight(epoch - 30)

#         model.eval()
#         ema_model.eval()
#         all_ls = []
#         all_ys = []
#         all_inputs = []
#         all_y_outputs = []
#         all_l_outputs = []

#         with torch.no_grad():
#             for i, (X, Y, _, T, ids, p) in enumerate(val_loader):
#                 # measure data loading time

#                 if self.config['gpu'] == None:
#                     X = X#.cuda()
#                     Y = Y#.cuda().float()
#                     T = T#.cuda().long()
#                 else:
#                     X = X.cuda()
#                     Y = Y.cuda().float()
#                     T = T.cuda().long()
#                 if self.config['dataset'] == 'mnist':
#                     X = X.view(X.shape[0], 1, -1)

#                 # compute output
#                 output = model(X)
#                 ema_output = ema_model(X)

#                 _, loss = criterion(output, Y)
#                 consistency_loss = consistency_weight * \
#                 consistency_criterion(output, ema_output) / X.shape[0]

#                 if self.check_mean_teacher(epoch):
#                     loss += consistency_loss
#                     predictions = torch.sign(ema_output).long()      
#                     output = ema_output
#                 else:
#                     predictions = torch.sign(output).long()
#                 all_ls.append(Y.cpu().numpy())
#                 all_ys.append(T.cpu().numpy())
#                 all_y_outputs.append(torch.sigmoid(output).cpu().numpy())
#         all_ys = np.concatenate(all_ys)
#         all_ls = np.concatenate(all_ls)
#         all_y_outputs = np.concatenate(all_y_outputs)
#         all_l_outputs = all_y_outputs * self.C
#         all_l_ests = all_y_outputs > 0.5 * self.C
#         all_y_ests = all_y_outputs > 0.5

#         # if set(np.unique(all_ls)) == {-1, 1}:
#         all_ys = (all_ys + 1) / 2
#         all_ls = (all_ls + 1) / 2
#         # elif not(set(np.unique(all_ls)).issubset({0, 1})):
#         #     raise ValueError(f"Labels should be either in {{-1, 1}} or {{0, 1}}, but {set(np.unique(all_ls))} is given.")

#         all_scores = self._calculate_validation_metrics(y_probs=all_y_outputs, y_vals=all_ys, l_probs=all_l_outputs, l_vals=all_ls, l_ests=all_l_ests, y_ests=all_y_ests)
#         all_scores['overall_loss'] = loss.detach().cpu().numpy() / (i + 1)
#         return all_scores
