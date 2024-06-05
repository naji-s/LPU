import copy
import logging
import unittest.mock
import types 


import chainer 
import chainer.training
import chainer.training
import chainer.training
import chainer.training
import chainer.training
import chainer.training
import chainer.training.extension
import chainer.variable
import chainer.utils.type_check
import numpy as np

import LPU.external_libs
import LPU.external_libs.nnPUlearning
import LPU.external_libs.nnPUlearning.pu_loss
import LPU.extras.nnPU_chainer


LOG = logging.getLogger(__name__)

# import torch.optim

import LPU.constants
import LPU.datasets.LPUDataset
import LPU.utils.dataset_utils
import LPU.utils.utils_general
import LPU.external_libs.nnPUlearning.train

LEARNING_RATE = 0.01
INDUCING_POINTS_SIZE = 32
TRAIN_VAL_RATIO = .1
HOLD_OUT_SIZE = None
TRAIN_TEST_RATIO = .5


def main():
    chainer.config.dtype = LPU.constants.NUMPY_DTYPE
    yaml_file_path = '/Users/naji/phd_codebase/LPU/configs/nnPU_chainer_config.yaml'
    config = LPU.utils.utils_general.load_and_process_config(yaml_file_path)
    lpu_dataset = LPU.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal', normalize=False, invert_l=False)
    train_loader, test_loader, val_loader, holdout_loader = LPU.utils.dataset_utils.create_stratified_splits(lpu_dataset, train_val_ratio=TRAIN_VAL_RATIO, batch_size=len(lpu_dataset), hold_out_size=HOLD_OUT_SIZE, train_test_ratio=TRAIN_TEST_RATIO)
    # passing X to initialize_inducing_points to extract the initial values of inducing points
    # nnPU_model = LPU.models.nnPU.nnPU(config)
    args = types.SimpleNamespace(**config)
    trainX, trainL, trainY = [item.detach().numpy() for item in next(iter(train_loader))]
    testX, testL, testY = [item.detach().numpy() for item in next(iter(test_loader))]
    valX, valL, valY = [item.detach().numpy() for item in next(iter(val_loader))]
    # converting {0, 1} to {-1, 1} for Y and L
    prior = np.mean(trainY)
    trainL = (2 * trainL - 1).tolist()
    testL = (2 * testL - 1).tolist()
    valL = (2 * valL - 1).tolist()
    trainY = (2 * trainY - 1).tolist()
    testY = (2 * testY - 1).tolist()
    valY = (2 * valY - 1).tolist()

    # Chainer implementation of nnPU and uPU assumes that the input data is 2D, 
    # so we need to add an extra dimension to the data if it is 1D
    if trainX[0].ndim == 1:
        trainX = [x[None, ...] for x in trainX]
        testX = [x[None, ...] for x in testX]
        valX = [x[None, ...] for x in valX]

    XYtrain = list(zip(trainX, trainL))
    XYtest = list(zip(testX, testY))

    dim = XYtrain[0][0].size // len(XYtrain[0][0])
    train_iter = chainer.iterators.SerialIterator(XYtrain, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(XYtrain, args.batchsize, repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(XYtest, args.batchsize, repeat=False, shuffle=False)

    loss_type = LPU.external_libs.nnPUlearning.train.select_loss(args.loss)
    # Model and iterator setup (as previously described)


    nnPU_model = LPU.extras.nnPU_chainer.nnPU(config, dim=dim)
    # trainer setup
    optimizers = {k: LPU.external_libs.nnPUlearning.train.make_optimizer(v, args.stepsize) for k, v in nnPU_model.models.items()}
    loss_funcs = {"nnPU": LPU.external_libs.nnPUlearning.pu_loss.PULoss(prior, loss=loss_type, nnpu=True, gamma=config['gamma'], beta=config['beta']),
                  "uPU": LPU.external_libs.nnPUlearning.pu_loss.PULoss(prior, loss=loss_type, nnpu=False)}

    nnPU_model.set_C(train_loader)
    # renew the train_iter after setting C
    train_loader = chainer.iterators.SerialIterator(train_loader, args.batchsize)
    for name, opt in optimizers.items():
        opt.setup(nnPU_model.models[name])
    nnPU_model.train_and_evaluate(train_iter, valid_iter, optimizers=optimizers, loss_funcs=loss_funcs, num_epochs=args.epoch)
    # Print some additional setup configurations to the console
    print("Training setup complete with the following configuration:")
    print(f"Batch size: {args.batchsize}, Epochs: {args.epoch}, Loss type: {args.loss}")

    print("prior: {}".format(prior))
    print("loss: {}".format(args.loss))
    print("batchsize: {}".format(args.batchsize))
    print("beta: {}".format(args.beta))
    print("gamma: {}".format(args.gamma))
    print("")

    # run training
    # trainer.run()


if __name__ == "__main__":
    with unittest.mock.patch.object(LPU.external_libs.nnPUlearning.pu_loss.PULoss, 'check_type_forward', lambda *x: True):
        main()

