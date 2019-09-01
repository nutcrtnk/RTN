import os.path
import numpy as np
import pandas as pd
import math
import pickle
import copy
import random
from torch.utils import data

import config


def unique_id(df, reindex=True):
    if reindex:
        _, val_u = pd.factorize(df['u'])
        _, val_i = pd.factorize(df['i'])
        idx_u = np.arange(len(val_u))
        idx_i = np.arange(len(val_i)) + config.shift_item_idx
    else:
        val_u = np.arange(df['u'].max()+1)
        val_i = np.arange(df['i'].max()+1)
        idx_u, idx_i = val_u, val_i
    dict_u = {v:k for k,v in zip(idx_u, val_u)}
    dict_i = {v:k for k,v in zip(idx_i, val_i)}

    if reindex:
        df['u'] = df['u'].map(dict_u)
        df['i'] = df['i'].map(dict_i)

    _map = dict()
    _map['to_new_uid'] = dict_u
    _map['to_new_iid'] = dict_i
    _map['to_old_uid'] = {v:k for k, v in dict_u.items()}
    _map['to_old_iid'] = {v:k for k, v in dict_i.items()}
    return df, _map


def remove_new_id(df_train, df_test, col='u'):
    a = df_train.groupby(col)['r'].count().to_frame()
    b = df_test.groupby(col)['t'].count().to_frame()
    c = a.merge(b, left_index=True, right_index=True, how='left')
    return df_test[df_test[col].isin(c.index)]


def split_train_test(df, n_split=1., n_min=1, dim='u', sort=True):
    if sort:
        df.sort_values(by=['t'], inplace=True)
    group_u = df.groupby(dim, as_index=False, group_keys=False)
    n_split = float(n_split)

    df_train = group_u.apply(lambda x: x[:len(x)-int(n_split)] if n_split.is_integer() else x[:-max(n_min, int(math.ceil(n_split*len(x))))])
    df_test = group_u.apply(lambda x: x[len(x)-int(n_split):] if n_split.is_integer() else x[-max(n_min, int(math.ceil(n_split*len(x)))):])
    return df_train, df_test


def load_dataset(dataset_name, sep=',', **kwargs):
    data_dir = config.rating_dir
    rating_df = pd.read_csv(os.path.join(data_dir, dataset_name + '.csv'), sep=sep, names=['u', 'i', 'r', 't'])
    rating_df, _map = unique_id(rating_df, **kwargs)
    return rating_df, _map


def df_to_dict(df):
    return df.groupby('u')['i'].apply(lambda x: x.tolist()).to_dict()


class Dataset:

    name = None

    train = None
    val = None
    train_val = None
    test = None

    df_train = None
    df_val = None
    df_train_val = None
    df_test = None

    n_users = 0
    n_items = 0
    n_actions = 0

    items = None
    users = None

    map = None

    _n_split = None
    _remove_new = None
    _shuffle = None

    @staticmethod
    def to_dict():
        d = dict()
        for k in dir(Dataset):
            if k[:2] != '__' and not callable(getattr(Dataset, k)):
                d[k] = getattr(Dataset, k)
        return d

    @staticmethod
    def load(name, n_split=1, sort=True, use_dump=True, remove_unseen=True, verbose=True, **kwargs):
        Dataset.name = name
        n_split = float(n_split)
        verboseprint = print if verbose else lambda *a, **k: None

        if use_dump:
            dump_file_name = os.path.join(config.dump_dir, name + '.pk')
            if os.path.exists(dump_file_name):
                verboseprint('load from dump')
                with open(dump_file_name, 'rb') as f:
                    for k, v in pickle.load(f).items():
                        setattr(Dataset, k, v)
                check_args = ('n_split', 'sort', 'remove_unseen')
                is_same = True
                for arg in check_args:
                    if getattr(Dataset, '_' + arg) != locals()[arg]:
                        verboseprint('different dump config %s=(args, dump): (%s, %s)' % (arg, str(locals()[arg]), str(getattr(Dataset, '_' + arg))))
                        is_same = False
                        break
                if is_same:
                    return

        verboseprint('preprocessing dataset')
        Dataset._n_split = float(n_split)
        Dataset._remove_unseen = remove_unseen
        Dataset._sort = sort

        df, Dataset.map = load_dataset(name, **kwargs)
        Dataset.df_train_val, Dataset.df_test = split_train_test(df, n_split=n_split, sort=sort)
        if n_split < 1:
            n_split = n_split / (1. - n_split)
        Dataset.df_train, Dataset.df_val = split_train_test(Dataset.df_train_val, n_split, sort=sort)
        Dataset.items = pd.factorize(Dataset.df_train['i'])[1]
        Dataset.users = pd.factorize(Dataset.df_train['u'])[1]

        if remove_unseen:
            mask_val = Dataset.df_val['i'].isin(Dataset.items)
            mask_test = Dataset.df_test['i'].isin(Dataset.items)
            mask_train_val = Dataset.df_train_val['i'].isin(Dataset.items)
            Dataset.df_val = Dataset.df_val[mask_val]
            Dataset.df_test = Dataset.df_test[mask_test]
            Dataset.df_train_val = Dataset.df_train_val[mask_train_val]

        Dataset.n_users = len(Dataset.map['to_new_uid'].keys())
        Dataset.n_items = len(Dataset.map['to_new_iid'].keys()) + config.shift_item_idx
        Dataset.n_actions = len(Dataset.df_train_val.index) + len(Dataset.df_test.index)

        Dataset.train = df_to_dict(Dataset.df_train)
        Dataset.val = df_to_dict(Dataset.df_val)
        Dataset.train_val = df_to_dict(Dataset.df_train_val)
        Dataset.test = df_to_dict(Dataset.df_test)

        if use_dump:
            dump_file_name = os.path.join(config.dump_dir, name + '.pk')
            with open(dump_file_name, 'wb') as f:
                pickle.dump(Dataset.to_dict(), f)

    def __init__(self, requires, name=None, process_mode=0, users=-1, timestep=-1,
                 n_neg=1):
        if name is None:
            assert Dataset.name is not None
        elif name != Dataset.name:
            self.load(name)

        self.name = name
        self.requires = requires
        self.timestep = timestep
        self.n_shift = 1
        self.n_neg = n_neg
        self.items = copy.deepcopy(Dataset.items)

        np.random.seed(config.random_seed)

        self.process_mode = process_mode

        self.test_users = set(Dataset.users)
        if process_mode == 1:
            self.data_dict = Dataset.train
            self.test_users = self.test_users.intersection(set(Dataset.val.keys()))
        elif process_mode == 2:
            self.data_dict = Dataset.train_val
            self.test_users = self.test_users.intersection(set(Dataset.test.keys()))
        else:
            self.data_dict = Dataset.train
        self.test_users = list(self.test_users)
        if isinstance(users, int):
            if users > 0:
                np.random.shuffle(self.test_users)
                self.test_users = self.test_users[:users]
        else:
            self.test_users = users

    def __len__(self):
        return len(self.test_users)

    def process(self, user):
        u_data = self.data_dict[user]
        if len(u_data) > self.timestep > 0:
            if not self.process_mode == 0:
                u_data = u_data[-self.timestep:]
            else:
                start = random.randint(0, len(u_data) - self.timestep - self.n_shift)
                u_data = u_data[start:start + self.timestep + self.n_shift]

        item_ids = np.array(u_data)
        pred_ids = None
        weight = None

        if 'pred_items' in self.requires:
            pos_pred_ids = np.roll(item_ids, -1)
            pos_pred_ids[len(pos_pred_ids) - 1:] = 0
            weight = np.array(pos_pred_ids > 0, dtype=np.float)
            weight = np.repeat(weight[np.newaxis, :], 1 + self.n_neg, 0)
            neg_pred_ids = np.random.choice(self.items, len(item_ids) * self.n_neg).reshape((self.n_neg, -1))
            pred_ids = np.concatenate([pos_pred_ids[np.newaxis, :], neg_pred_ids])

        return item_ids, pred_ids, weight

    def __getitem__(self, index):
        user = self.test_users[index]
        items, pred_items, weight = self.process(user)
        X = {}
        if 'users_items' in self.requires:
            X['users_items'] = np.repeat(items[np.newaxis, :], 1 + self.n_neg, 0)
        if 'pred_items' in self.requires:
            X['pred_items'] = pred_items
        if 'users' in self.requires:
            X['users'] = np.ones(1 + self.n_neg, dtype=np.int) * user
        return X, weight


def pad(array, size, dim=1, pad_front=True):
    if size - array.shape[dim] > 0:
        pad_size = [(0, 0) for _ in range(len(array.shape))]
        pad_size[dim] = (size - array.shape[dim], 0) if pad_front else (0, size - array.shape[dim])
        return np.pad(array, pad_width=pad_size, mode='constant')
    return array


def tbptt(inputs, timestep, unroll):
    outputs = []
    for i in range(0, timestep, unroll):
        output = []
        for input in inputs:
            if input is None:
                outputs.append(None)
            elif isinstance(input, dict):
                output.append({k: x if len(x.shape) == 1 else pad(x[:,i:i+unroll], unroll) for k, x in input.items()})
            elif isinstance(input, list):
                output.append([x if len(x.shape) == 1 else pad(x[:, i:i + unroll], unroll) for x in input])
            else:
                x = input
                output.append(x if len(x.shape) == 1 else pad(x[:, i:i + unroll], unroll))
        outputs.append(output)
    return outputs


class CustomCollate:

    def __init__(self, unroll=-1, timestep_dim=1):
        self.unroll = unroll
        self.timestep_dim = timestep_dim

    def __call__(self, batch):
        x_0 = batch[0][0]
        new_x = {}
        new_w = []
        max_timestep = 0
        for k in x_0.keys():
            new_x[k] = []
        count = 0
        for x, w in batch:
            count += 1
            for k, v in x.items():
                new_x[k].append(v)
                if len(v.shape) > self.timestep_dim:
                    max_timestep = max(max_timestep, v.shape[self.timestep_dim])
            if w is not None:
                new_w.append(w)
                max_timestep = max(max_timestep, w.shape[self.timestep_dim])
        # padding
        for k, x in new_x.items():
            for i, _x in enumerate(x):
                if len(_x.shape) > self.timestep_dim:
                    x[i] = pad(_x, max_timestep, self.timestep_dim)
            new_x[k] = np.concatenate(x)
        new_w = np.concatenate(list(map(lambda x: pad(x, max_timestep, self.timestep_dim), new_w))) if len(new_w) > 0 else None

        output = [new_x, new_w]
        if self.unroll > 0:
            output = tbptt(output, timestep=max_timestep, unroll=self.unroll)
        else:
            output = [output]
        return output


def get_loader(requires, batch_size=100, unroll=-1, num_workers=4, shuffle=True, **kwargs):
    ds = Dataset(requires, **kwargs)
    return data.DataLoader(ds, batch_size=batch_size, collate_fn=CustomCollate(unroll=unroll), num_workers=num_workers, shuffle=shuffle)
