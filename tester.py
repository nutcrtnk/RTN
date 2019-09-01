import numpy as np
import torch
import inspect
from collections import OrderedDict

from data_loader import Dataset, get_loader
import config


def random_candidates(test_users, n_candidates=101, seed=37, validate=False):
    if not validate:
        np.random.seed(seed)
        train_data = Dataset.train_val
        test_data = Dataset.test
    else:
        train_data = Dataset.train
        test_data = Dataset.val

    test_set = {}
    for i, user in enumerate(test_users):
        target = test_data[user]
        user_items = train_data[user]
        items = np.random.choice(Dataset.items, len(target) + len(user_items) + n_candidates, replace=False)
        negs = set(items) - set(user_items) - set(target)
        test_set[user] = list(negs)[:n_candidates - len(target)] + target
    return test_set


class Evaluator:

    def __init__(self, model, k=50, users=-1, n_candidates=-1, timestep=-1, metrics='recall,ndcg', validate=False, repeatable=False):
        self.model = model
        self.k = k
        self.n_candidates = n_candidates
        self.instances = []
        self.test_users = []
        self.metrics = metrics.split(',')
        self.validate = validate
        self.rtn_timestep = timestep
        self.generator = self.create_generator(users)
        self.repeatable = repeatable
        self.invalid_items = list(set(range(Dataset.n_items)) - set(Dataset.items))

    def add(self, target, prediction):
        self.instances.append((target, prediction))

    def create_generator(self, users):
        requires = set(inspect.signature(self.model.predict).parameters) | {'users', 'users_items'}
        gen_params = dict()
        gen_params['n_neg'] = 0
        gen_params['unroll'] = -1
        gen_params['timestep'] = -1
        return get_loader(requires, batch_size=100, users=users, num_workers=2, process_mode=1 if self.validate else 2, shuffle=False, **gen_params)

    def precision(self):
        prec = 0
        for target, prediction in self.instances:
            prec += float(len(set(target) & set(prediction))) / len(prediction)
        return prec / len(self.instances)

    def recall(self):
        recall = 0
        count = 0
        for target, prediction in self.instances:
            rec = float(len(set(target) & set(prediction))) / len(target)
            count += 1
            recall += rec
        return recall / count

    def ndcg(self):
        ndcg_ = 0.
        count = 0
        for target, prediction in self.instances:
            dcg = 0.
            max_dcg = 0.
            for i, p in enumerate(prediction):
                if i < len(target):
                    max_dcg += 1. / np.log2(2 + i)
                if p in target:
                    dcg += 1. / np.log2(2 + i)
            ndcg_ += dcg / max_dcg
            count += 1
        return ndcg_ / count

    def filter_topk(self, score, users_items, candidates, invalid_items=None):
        if candidates is None:
            if invalid_items is not None:
                score[:, invalid_items] = -float('inf')
            if not self.repeatable:
                score = score.scatter_(1, users_items, -float('inf'))
        scores, topk_items = score.topk(self.k, -1)
        if candidates is not None:
            topk_items = candidates.gather(1, topk_items)
        # print('=============================')
        # print(users_items[0], topk_items[0])
        return topk_items

    def run(self, calculate=True):
        self.instances = []
        self.test_users = []
        self.model.eval()

        device = next(self.model.parameters()).device

        test_data = Dataset.test
        if self.validate:
            test_data = Dataset.val

        with torch.no_grad():
            invalid_items = torch.LongTensor(self.invalid_items).to(device)
            for inputs in self.generator:
                # inputs -> [multi-batch] multi-batch-> [X, Y, W]
                inp = inputs[0][0]
                users = inp['users']
                if self.n_candidates <= 0:
                    candidates = None
                else:
                    candidates = []
                    test_case = random_candidates(users, self.n_candidates, seed=config.random_seed, validate=self.validate)
                    for user in users:
                        candidates.append(test_case[user])
                    candidates = torch.LongTensor(candidates).to(device)
                inp = {k: torch.LongTensor(x).to(device) for k, x in inp.items()}
                users_items = inp['users_items'].clone()
                if self.rtn_timestep > 0:
                    inp['users_items'] = inp['users_items'][:, -self.rtn_timestep:]
                score, hidden = self.model.predict(candidates=candidates, **inp)
                prediction = self.filter_topk(score.detach(), users_items=users_items, candidates=candidates, invalid_items=invalid_items)
                prediction = prediction.cpu().numpy()
                users_items = users_items.cpu().numpy()
                for i, user in enumerate(users):
                    target = test_data[user]
                    # target = set(users_items[i])
                    self.test_users.append(user)
                    self.add(target, prediction[i])

        if calculate:
            return self.calculate_metrics()

    def calculate_metrics(self):
        return OrderedDict([(metric, getattr(self, metric)()) for metric in self.metrics])
