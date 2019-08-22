'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from utility.helper import *

class Data(object):
    def __init__(self, path, batch_size, dataset, neg_num):
        self.path = path
        self.batch_size = batch_size
        self.dataset = dataset
        self.neg_num = neg_num

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
		#总user/item
        self.n_train, self.n_test = 0, 0
		#非空即user-item interaction数目
        self.neg_pools = {}

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                    #R为交互矩

                    self.train_items[uid] = train_items
					#Ru

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items
					#Ru

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz('/output/%s/mat/%s/%s.npz' % (self.neg_num , self.dataset, 's_adj_mat'))
            norm_adj_mat = sp.load_npz('/output/%s/mat/%s/%s.npz' % (self.neg_num , self.dataset, 's_norm_mat'))
            mean_adj_mat = sp.load_npz('/output/%s/mat/%s/%s.npz' % (self.neg_num , self.dataset, 's_mean_mat'))
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            ensureDir('/output/%s/mat/%s/%s.npz' % (self.neg_num , self.dataset, 's_adj_mat'))
            sp.save_npz('/output/%s/mat/%s/%s.npz' % (self.neg_num , self.dataset, 's_adj_mat'), adj_mat)
            sp.save_npz('/output/%s/mat/%s/%s.npz' % (self.neg_num , self.dataset, 's_norm_mat'), norm_adj_mat)
            sp.save_npz('/output/%s/mat/%s/%s.npz' % (self.neg_num , self.dataset, 's_mean_mat'), mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
		#链表稀疏矩阵lil
        R = self.R.tolil()

		#矩阵总大小是(n+m)*(n+m)的，这两步相当于eq(8)，亦即A矩阵
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
		#Convert this matrix to Dictionary Of Keys format.
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

		#使矩阵变成单位矩
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
			#(n+m)*1矩阵，每行为原矩阵相应行之和
            d_inv = np.power(rowsum, -1).flatten()
			#每个数取倒并排列为行向量
            d_inv[np.isinf(d_inv)] = 0.
			#原本为零取倒为inf，这里对inf进行赋零
            d_mat_inv = sp.diags(d_inv)
			#建立稀疏的对角rowsum可能很多)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()
			#Convert this matrix to Coordinate format.一种稀疏矩阵存储方

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        #adj_mat <=> plain
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		#norm
        mean_adj_mat = normalized_adj_single(adj_mat)
		#gcmc

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
		#Convert this matrix to Compressed Sparse Row format.很巧妙的一种存储方

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample_pos_items_for_u1(u, num):
        pos_items = self.train_items[u]
        n_pos_items = len(pos_items)
        pos_batch = []
        while True:
            if len(pos_batch) == num: break
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]

            if pos_i_id not in pos_batch:
                pos_batch.append(pos_i_id)
        return pos_batch

    def sample(self, ratings, neg_num, users):


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, X):
            neg_items = []
            while True:
                if len(neg_items) == X: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
        
            return neg_items
        

        pos_items, neg_items = [], []
        for i in range(len(users)):
            u = users[i]
            items_for_u_temp = sample_neg_items_for_u(u, neg_num)
            item_dict = {}

            #比直接比较更快
            for j in items_for_u_temp:
                item_dict[j] = ratings[i][j]

            item_to_add = max(item_dict, key = item_dict.get)  

            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += [item_to_add]
            

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open('/output/%s/%s/sparsity.split' % (self.neg_num , self.dataset), 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            ensureDir('/output/%s/%s/sparsity.split' % (self.neg_num , self.dataset))
            f = open('/output/%s/%s/sparsity.split' % (self.neg_num , self.dataset), 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state


	#创建四个不同sparsity的集
    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
		#总interaction
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
			#a={1:2,5:231}
			#list(enumerate(a))
			#>[(0,1),(1,5)]
            temp += user_n_iid[n_iids]
			#长度*个数
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
