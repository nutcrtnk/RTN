import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import inspect

from tester import Evaluator
import os
from data_loader import get_loader, Dataset

import config


def ema_loss(losses, *args):
    w = 0.1
    new_loss = to_numpy(*args)
    if losses is None:
        losses = new_loss
    else:
        losses = losses * (1 - w) + new_loss * w
    return losses


def to_numpy(*args):
    return np.array([arg.item() for arg in args])


class Trainer:

    def __init__(self, model, out_name='RTN.pt', metric=0):
        assert Dataset.name is not None, 'Please call Dataset.load($dataset_name)'
        self.model = model
        self.out_name = out_name

        self.optimizer = None
        self.optimizer_name = ''
        self.optimizer_kwargs = None
        self.clip = 0

        self.dataset_name = Dataset.name
        self.epochs = 0
        self.best_result = None
        self.device = torch.device('cpu')
        self.evaluator = Evaluator(model)
        self.eval_criterion = metric
        self.generator = None

        self._losses = []
        self._batch_sizes = []
        self._hidden = None
        self._iter_time = None
        self._from_best_epochs = 0
        self._last_eval_epochs = 0
        self._requires = [k for k in inspect.signature(self.model.forward).parameters]

    def cuda(self, device_id=None):
        if device_id is not None:
            torch.cuda.set_device(device_id)
        self.model.cuda()
        self.device = 'cuda'

    def set_optimizer(self, optim_name='Adam', clip=0, **kwargs):
        self.optimizer_name = optim_name
        self.optimizer_kwargs = kwargs
        self.optimizer = getattr(optim, optim_name)(filter(lambda p: p.requires_grad, self.model.parameters()), **kwargs)
        self.clip = clip

    def train_one_iter(self, inp, is_new_sequence=True):
        np.set_printoptions(precision=4)

        if self._iter_time is None:
            self._iter_time = time.time()

        self.optimizer.zero_grad()
        _t_gen = time.time()
        inputs, weight = inp
        batch_size = len(weight)
        inputs = {k: torch.LongTensor(x).to(self.device) for k, x in inputs.items()}
        weight = torch.Tensor(weight).to(self.device)

        self.model.train()
        if is_new_sequence:
            self._hidden = None
        outputs = self.model(**inputs, hidden=self._hidden, weight=weight)
        self._hidden = outputs[-1].detach()

        out_losses = outputs[:-1]
        out_losses = [out_loss.sum() for out_loss in out_losses]
        loss = sum(out_losses)
        self._losses.append(to_numpy(loss, *out_losses))
        self._batch_sizes.append(batch_size)

        loss.backward()
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

    def increment_step(self, verbose=1):
        self.epochs += 1
        if verbose == 1:
            loss = (np.stack(self._losses, axis=-1) * np.array(self._batch_sizes)).sum(axis=-1) / sum(self._batch_sizes)
            print('TRAIN LOSS:', loss, 'TIME: %.4f' % (time.time() - self._iter_time))
            self._iter_time = time.time()

    def eval(self, verbose=1, mode='max'):
        np.set_printoptions(precision=4)
        t1 = time.time()
        result = self.evaluator.run()
        eval_time = time.time() - t1

        metric = self.eval_criterion
        if isinstance(metric, int):
            metric = list(result.keys())[metric]
        is_best = self.best_result is None
        if not is_best:
            if mode == 'min':
                is_best = result[metric] < self.best_result[metric]
            if mode == 'max':
                is_best = result[metric] > self.best_result[metric]
        if is_best:
            self.best_result = result
            self._from_best_epochs = 0
            self.save()
        else:
            self._from_best_epochs += self.epochs - self._last_eval_epochs
        self._last_eval_epochs = self.epochs

        if verbose == 1:
            self.print_eval(result, eval_time=eval_time)
        if (verbose == 2) and is_best:
            self.print_eval(result, epochs=self.epochs)

    def train(self, epochs=10000, min_epochs=0, batch_size=100, eval_step=5, timestep=50, verbose=1,
              patience=10, num_workers=4, reset=True):
        assert self.optimizer is not None, 'Optimizer have not been set'
        assert self.evaluator is not None, 'Evaluator have not been set'

        np.set_printoptions(precision=4)

        if reset:
            self.model.reset()

        train_gen = get_loader(self._requires, batch_size=batch_size, num_workers=num_workers,
                               process_mode=0, timestep=timestep, n_neg=self.model.n_neg)
        self.generator = train_gen

        print('start training')
        start_t = time.time()
        gen_time = 0
        train_time = 0

        # #begin training
        last_epoch = -1
        while True:
            if verbose == 1 and (self.epochs != last_epoch):
                print('STEP: ', self.epochs + 1)
                print('GEN TIME:', gen_time)
                print('TRAIN TIME:', train_time)
                gen_time = 0
                train_time = 0
                last_epoch = self.epochs
            t1 = time.time()
            self._losses, self._batch_sizes = [], []
            for inputs in train_gen:
                for i, inp in enumerate(inputs):
                    gen_time += time.time() - t1
                    t1 = time.time()
                    self.train_one_iter(inp, is_new_sequence=(i == 0))
                    train_time += time.time() - t1
                    t1 = time.time()
            self.increment_step(verbose=verbose)
            if self.epochs % eval_step == 0:
                self.eval(verbose=verbose)
                if self.epochs >= min_epochs:
                    if patience > 0 and (self._from_best_epochs >= patience):
                        break
            if self.epochs == epochs:
                break

        self.print_eval(self.best_result, self.epochs, print_dataset=True,
                        eval_time=time.time() - start_t, title='Training result', flush=True)

    def print_eval(self, result, epochs=None, print_dataset=False, eval_time=None, title=None,
                   flush=False):
        print('---------------------------------------')
        if title is not None:
            print(title)
        if print_dataset:
            print(self.dataset_name)
        if epochs is not None:
            print('STEP', epochs)
        print('EVAL', end=' ')
        for eval_crit in sorted(result):
            print('%s: %.4f,' % (eval_crit, result[eval_crit]), end=' ')
        if eval_time is not None:
            print('TIME: %.4f,' % eval_time)
        else:
            print('')
        print('---------------------------------------', flush=flush)

    def get_filedir(self):
        return config.model_dir / self.dataset_name / self.out_name

    def save(self, verbose=0):
        out_dir = self.get_filedir()
        os.makedirs(out_dir.parent, exist_ok=True)
        self.model.save(out_dir)
        if verbose:
            print('save file to {}'.format(out_dir))
