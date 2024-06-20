"""
Implementation of the nnPU model and uPU model by inheritance from the LPUModelBase class.
Despite this, becasue he code uses Chainer, the details of traininig are not adapted other
pytorch models in this library. That is a future TODO (naji)
"""
import copy
import logging
import random


import chainer
import chainer.training
import chainer.training.extensions
import chainer.functions
import numpy as np
import sklearn.metrics
import sklearn.model_selection
import six

import LPU.constants
import LPU.external_libs
import LPU.extras.dedpul_by_mpe
import LPU.models.geometric.Elkan.Elkan
import LPU.models.lpu_model_base
import LPU.external_libs.nnPUlearning.train

LOG = logging.getLogger(__name__)

EPSILON = 1e-16


EPSILON = 1e-16
import six
import copy
import numpy as np
from chainer import report
from chainer import Variable
from sklearn.metrics import roc_auc_score
import chainer.functions as F
import chainer


import chainer
from chainer import training, Variable, optimizers, iterators, serializers, functions as F
from chainer.training import extensions
import numpy as np
import copy

import LPU.models.lpu_model_base as base
import LPU.external_libs.nnPUlearning.train
import LPU.external_libs.nnPUlearning.pu_loss

class ProbabilisticLGivenXEvaluator(chainer.training.extensions.Evaluator):

    def __init__(self, *args, **kwargs):
        super(ProbabilisticLGivenXEvaluator, self).__init__(*args, **kwargs)

    def evaluate(self):
        iterator = self._iterators['main']
        targets = self.get_all_targets()

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = chainer.reporter.DictSummary()
        for batch in it:
            observation = {}
            with chainer.reporter.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                if isinstance(in_arrays, tuple):
                    in_vars = tuple(Variable(x)
                                    for x in in_arrays)
                    for k, target in targets.items():
                        target.compute_s_given_X(*in_vars)
                elif isinstance(in_arrays, dict):
                    in_vars = {key: Variable(x)
                               for key, x in six.iteritems(in_arrays)}
                    for k, target in targets.items():
                        target.compute_s_given_X(**in_vars)
                else:
                    in_vars = Variable(in_arrays)
                    for k, target in targets.items():
                        target.compute_s_given_X(in_vars)
            summary.add(observation)

        return summary.compute_mean()
    
class ProbabilisticYGivenXEvaluator(chainer.training.extensions.Evaluator):

    def __init__(self, *args, **kwargs):
        super(ProbabilisticYGivenXEvaluator, self).__init__(*args, **kwargs)

    def evaluate(self):
        iterator = self._iterators['main']
        targets = self.get_all_targets()

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = chainer.reporter.DictSummary()
        for batch in it:
            observation = {}
            with chainer.reporter.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                if isinstance(in_arrays, tuple):
                    in_vars = tuple(Variable(x)
                                    for x in in_arrays)
                    for k, target in targets.items():
                        target.compute_y_given_X(*in_vars)
                elif isinstance(in_arrays, dict):
                    in_vars = {key: Variable(x)
                               for key, x in six.iteritems(in_arrays)}
                    for k, target in targets.items():
                        target.compute_y_given_X(**in_vars)
                else:
                    in_vars = Variable(in_arrays)
                    for k, target in targets.items():
                        target.compute_y_given_X(in_vars)
            summary.add(observation)

        return summary.compute_mean()    



class nnPU(base.LPUModelBase):
    def __init__(self, config, dim=None, gpu=-1, **kwargs):
        super(nnPU, self).__init__(**kwargs)
        self.config = config
        self.prior = config.get('prior', 0.5)
        self.dim = dim
        self.setup_models()
        self.gpu = gpu
        if gpu >= 0:
            chainer.backends.cuda.get_device_from_id(self.gpu).use()
            for model in self.models.values():
                model.to_gpu()

    def select_model(self, model_name):
        models = {
            "linear": LPU.external_libs.nnPUlearning.model.LinearClassifier, 
            "3lp": LPU.external_libs.nnPUlearning.model.ThreeLayerPerceptron,
            "mlp": LPU.external_libs.nnPUlearning.model.MultiLayerPerceptron, 
            "cnn": LPU.external_libs.nnPUlearning.model.CNN
        }
        return models[model_name](self.prior, self.dim)
    
    def set_C(self, holdout_dataloader):
        assert len(holdout_dataloader) == 1
        X, holdout_l, holdout_y = next(iter(holdout_dataloader))
        self.C = holdout_l[holdout_y == 1].mean()


    def setup_models(self):
        model = self.select_model(self.config['model'])
        self.models = {
            'nnPU': model,
            'uPU': copy.deepcopy(model)
        }

    def train_and_evaluate(self, train_iter, valid_iter= None, loss_funcs=None, optimizers=None, num_epochs=100):
        updater = LPU.external_libs.nnPUlearning.train.MultiUpdater(train_iter, optimizers, self.models, device=self.gpu, loss_func=loss_funcs)
        trainer = chainer.training.Trainer(updater, stop_trigger=(num_epochs, 'epoch'), out=self.config['out'])
        # trainer.extend(chainer.training.extensions.LogReport(trigger=(1, 'epoch')))
        # trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'nnPU/loss', 'uPU/loss', 'elapsed_time']), trigger=(1, 'epoch'))
        log_report = chainer.training.extensions.LogReport(trigger=(1, 'epoch'))

        # Evaluators to compute metrics on training, testing, and validation datasets
        train_01_loss_evaluator = LPU.external_libs.nnPUlearning.train.MultiPUEvaluator(self.prior, valid_iter, self.models, device=self.gpu)
        train_01_loss_evaluator.default_name = 'trainPU'

        prob_y_given_X_valid_evaluator = ProbabilisticYGivenXEvaluator(valid_iter, self.models, device=self.gpu)
        prob_y_given_X_valid_evaluator.default_name = 'y_X_valid_eval_auc'
        train_iter_copy = copy.deepcopy(train_iter)
        train_iter_copy.reset()
        train_iter_copy._repeat = False
        prob_l_given_X_train_evaluator = ProbabilisticLGivenXEvaluator(train_iter_copy, self.models, device=self.gpu)
        prob_l_given_X_train_evaluator.default_name = 'l_X_train_eval_auc'


        multi_valid_evaluator = LPU.external_libs.nnPUlearning.train.MultiEvaluator(valid_iter, self.models, device=self.gpu)
        multi_valid_evaluator.default_name = 'valid_eval'

        # Extensions are added in the order of their execution and logging output
        trainer.extend(log_report)
        trainer.extend(train_01_loss_evaluator, trigger=(1, 'epoch'))
        trainer.extend(prob_l_given_X_train_evaluator, trigger=(1, 'epoch'))
        trainer.extend(prob_y_given_X_valid_evaluator, trigger=(1, 'epoch'))
        trainer.extend(multi_valid_evaluator, trigger=(1, 'epoch'))
        trainer.extend(chainer.training.extensions.PrintReport(
        [
            'epoch',  # Current epoch number
            'trainPU/nnPU/error', 'trainPU/uPU/error',  # Training errors for nnPU and uPU
            'l_X_train_eval_auc/nnPU/score', 'l_X_train_eval_auc/uPU/score',    # AUC for uPU on train and test sets
            'y_X_valid_eval_auc/nnPU/score', 'y_X_valid_eval_auc/uPU/score',  # Additional metrics from MultiEvaluator
            'valid_eval/nnPU/error', 'valid_eval/uPU/error',  # Additional metrics from MultiEvaluator
            'elapsed_time'  # Total elapsed time
        ]),
        trigger=(1, 'epoch')
    )

        # # In case you want to visualize your training errors and accuracy graphically using a plot report
        if chainer.training.extensions.PlotReport.available():
                trainer.extend(
                    chainer.training.extensions.PlotReport(['trainPU/nnPU/error', 'trainPU/uPU/error'], 'epoch', file_name='training_error.png'))
                trainer.extend(
                    chainer.training.extensions.PlotReport(['valid_eval/nnPU/error', 'valid_eval/uPU/error'], 'epoch', file_name='valid_error.png'))

        trainer.run()

    def predict_prob_y_given_X(self, X, model_to_use='nnPU'):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            X_var = Variable(X)
            predictions = {name: model.calculate(X_var) for name, model in self.models.items()}
            probabilities = {name: F.softmax(pred).data for name, pred in predictions.items()}
        return probabilities[model_to_use]

    def predict_prob_l_given_y_X(self, X):
        return self.C
    
    def predict_prob_l_given_X(self, X):
        return self.predict_prob_y_given_X(X) * self.C



if __name__ == "__main__":
    config = {
        'model': '3lp',
        'loss': 'sigmoid',
        'gamma': 0.2,
        'beta': 1.0,
        'stepsize': 0.01,
        'out': './scripts/nnPU/checkpoints',
        'gpu': 0,
        'dataset': 'mnist',
        'labeled': 1000,
        'unlabeled': 59000,
        'batchsize': 256
    }
    loss_type = LPU.external_libs.nnPUlearning.train.select_loss(config['loss'])
    XYtrain, XYtest, prior = LPU.external_libs.nnPUlearning.dataset.load_dataset(config['dataset'], config['labeled'], config['unlabeled'])
    loss_funcs = {"nnPU": LPU.external_libs.nnPUlearning.pu_loss.PULoss(prior, loss=loss_type, nnpu=True, gamma=config['gamma'], beta=config['beta']),
                  "uPU": LPU.external_libs.nnPUlearning.pu_loss.PULoss(prior, loss=loss_type, nnpu=False)}
    
    dim = XYtrain[0][0].size // len(XYtrain[0][0])
    model = nnPU(config, prior=prior, dim=dim, loss_funcs=loss_funcs)
    train_iter = chainer.iterators.SerialIterator(XYtrain, batch_size=config['batchsize'],  repeat=True)
    optimizers = {name: optimizers.Adam() for name in model.models.keys()}
    for name, opt in optimizers.items():
            opt.setup(model.models[name])
    print("Starting training...")
    model.train(train_iter, optimizers, num_epochs=1)        
