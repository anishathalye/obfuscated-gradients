from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.append("..")
import os
import argparse
import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.metrics import accuracy_score
from util import (random_split, block_split, train_lr, compute_roc)

DATASETS = ['mnist', 'cifar', 'svhn']
ATTACKS = ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l0', 'cw-l2']
ARTIFACTS = ['kd', 'bu', 'lid']
PATH_DATA = "../data/"
# PATH_DATA = "../data_v1/"
PATH_IMAGES = "../plots/"

def load_artifacts(dataset, attack, artifacts):
    """
    Load multiple artifacts for one dataset and one attack.
    :param dataset: 
    :param attack: 
    :param artifacts: 
    :return: 
    """
    X, Y = None, None
    for artifact in artifacts:
        print("  -- %s" % artifact)
        #file_name = os.path.join(PATH_DATA, "%s_%s_%s.npy" % (artifact, dataset, attack))
        file_name = os.path.join("data/lid_cifar_cw-l2_20.npy")
        data = np.load(file_name)
        if X is None:
            X = data[:, :-1]
        else:
            X = np.concatenate((X, data[:, :-1]), axis=1)
        if Y is None:
            Y = data[:, -1] # labels only need to load once

    return X, Y

def detect(args):
    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    # assert args.attack in ATTACKS, \
    #     "Training attack must be either 'fgsm', 'bim-a', 'bim-b', " \
    #     "'jsma', 'cw-l0, cw-l2 or cw-li'"
    # if args.test_attack is not None:
    #     assert args.test_attack in ATTACKS, \
    #         "Test attack must be either 'fgsm', 'bim-a', 'bim-b', " \
    #         "'jsma', 'cw-l0, cw-l2 or cw-li'"
    artifacts = args.artifacts.split(',')
    for atf in artifacts:
        assert atf in ARTIFACTS, \
            "Artifact(s) to use 'kd', 'bu', 'lid'"

    X, Y = load_artifacts(args.dataset, args.attack, artifacts)
    #X = X[:, :-1]

    # normalization
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    # X = scale(X) # Z-norm

    if args.test_attack is None or args.test_attack == args.attack:
        # test attack is the same as training attack
        X_train, Y_train, X_test, Y_test = block_split(X, Y)
    else:
        # test attack is a different attack
        # load the test attack for testing
        print("Loading test attack: %s" % args.test_attack)
        X_train, Y_train = X, Y
        X_test, Y_test = load_artifacts(args.dataset, args.test_attack, artifacts)
        # using training normalizer
        X_test = scaler.transform(X_test)
        # X_test = scale(X_test) # Z-norm

        # for adversarial testing - only use adversarial examples:
        num_samples = X_test.shape[0]
        partition = int(num_samples / 3)
        # X_test, Y_test = X_test[:partition], Y_test[:partition] # only test the accuracy on the advs
        X_test, Y_test = X_test[:2*partition], Y_test[:2*partition] # test accuracy on normal and advs
        # X_noisy, Y_noisy = X_test[2 * partition:], Y_test[2*partition:]

    # test -- use one layer of lid
    # X_train = X_train[:, -1, None]
    # X_test = X_test[:, -1, None]

    print("Train samples size: ", X_train.shape)
    print("Test samples size: ", X_test.shape)

    # print(X_train[0])
    # print(X_test[0])

    ## Build detector
    print("LR Detector on [dataset: %s, train_attack: %s, test_attack: %s] with:" %
                                        (args.dataset, args.attack, args.test_attack))
    lr = train_lr(X_train, Y_train)

    ## Evaluate detector
    # Compute logistic regression model predictions
    y_pred = lr.predict_proba(X_test)[:, 1]
    # Compute AUC
    n_samples = len(X_test)
    _, _, auc_score = compute_roc(Y_test, y_pred)

    y_label_pred = lr.predict(X_test)
    acc = accuracy_score(Y_test, y_label_pred)
    print('Detector ROC-AUC score: %0.4f, accuracy: %.4f' % (auc_score, acc))

    return lr, auc_score, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use train the discriminator; either 'fgsm', 'bim-a', 'bim-b', 'jsma' 'cw' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-r', '--artifacts',
        help="Artifact(s) to use any combination in ['kd', 'bu', 'lid'] "
             "separated by comma, for example: kd,bu",
        required=True, type=str
    )
    parser.add_argument(
        '-t', '--test_attack',
        help="Artifact(s) to cross-test the discriminator. This is to test the detection "
             "power of the discriminator trained over one attack on adversarials generated "
             "by the other attack",
        required=False, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=100)
    parser.set_defaults(test_attack=None)
    # args = parser.parse_args()
    # PATH_DATA = '../data_v1/'
    #args = parser.parse_args(['-d', 'mnist', '-a', 'fgsm', '-t', 'fgsm', '-r', 'kd,bu'])
    args = parser.parse_args(['-d', 'mnist', '-a', 'cw-l2', '-t', 'fgsm', '-r', 'lid'])
    detect(args)
