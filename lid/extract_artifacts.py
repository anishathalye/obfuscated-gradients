from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
from keras.models import load_model

import sys
sys.path.append("..")

from util import (get_data, get_noisy_samples, get_mc_predictions,
                      get_deep_representations, score_samples, normalize,
                      get_lids_random_batch, get_kmeans_random_batch)

# In the original paper, the author used optimal KDE bandwidths dataset-wise
#  that were determined from CV tuning
BANDWIDTHS = {'mnist': 3.7926, 'cifar': 0.26, 'svhn': 1.00}

# Here we further tune bandwidth for each of the 10 classes in mnist, cifar and svhn
# Run tune_kernal_density.py to get the following settings.
# BANDWIDTHS = {'mnist': [0.2637, 0.1274, 0.2637, 0.2637, 0.2637, 0.2637, 0.2637, 0.2069, 0.3360, 0.2637],
#               'cifar': [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#               'svhn': [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1274, 0.1000, 0.1000]}

PATH_DATA = "../data/"
# PATH_DATA = "../data_v1/"
PATH_IMAGES = "../images_v1/"

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("X_pos: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("X_neg: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y


def get_kd(model, X_train, Y_train, X_test, X_test_noisy, X_test_adv):
    """
    Get kernel density scores
    :param model: 
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param X_test_noisy: 
    :param X_test_adv: 
    :return: artifacts: positive and negative examples with kd values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    # Get deep feature representations
    print('Getting deep feature representations...')
    X_train_features = get_deep_representations(model, X_train,
                                                batch_size=args.batch_size)
    X_test_normal_features = get_deep_representations(model, X_test,
                                                      batch_size=args.batch_size)
    X_test_noisy_features = get_deep_representations(model, X_test_noisy,
                                                     batch_size=args.batch_size)
    X_test_adv_features = get_deep_representations(model, X_test_adv,
                                                   batch_size=args.batch_size)
    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(Y_train.shape[1]):
        class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]
    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined "
                  "optimal for the specific CNN models of the paper. If you've "
                  "changed your model, you'll need to re-optimize the "
                  "bandwidth.")
    print('bandwidth %.4f for %s' % (BANDWIDTHS[args.dataset], args.dataset))
    for i in range(Y_train.shape[1]):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=BANDWIDTHS[args.dataset]) \
            .fit(X_train_features[class_inds[i]])
    # Get model predictions
    print('Computing model predictions...')
    preds_test_normal = model.predict_classes(X_test, verbose=0,
                                              batch_size=args.batch_size)
    preds_test_noisy = model.predict_classes(X_test_noisy, verbose=0,
                                             batch_size=args.batch_size)
    preds_test_adv = model.predict_classes(X_test_adv, verbose=0,
                                           batch_size=args.batch_size)
    # Get density estimates
    print('computing densities...')
    densities_normal = score_samples(
        kdes,
        X_test_normal_features,
        preds_test_normal
    )
    densities_noisy = score_samples(
        kdes,
        X_test_noisy_features,
        preds_test_noisy
    )
    densities_adv = score_samples(
        kdes,
        X_test_adv_features,
        preds_test_adv
    )

    print("densities_normal:", densities_normal.shape)
    print("densities_adv:", densities_adv.shape)
    print("densities_noisy:", densities_noisy.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # densities_normal_z, densities_adv_z, densities_noisy_z = normalize(
    #     densities_normal,
    #     densities_adv,
    #     densities_noisy
    # )

    densities_pos = densities_adv
    densities_neg = np.concatenate((densities_normal, densities_noisy))
    artifacts, labels = merge_and_generate_labels(densities_pos, densities_neg)

    return artifacts, labels

def get_bu(model, X_test, X_test_noisy, X_test_adv):
    """
    Get Bayesian uncertainty scores
    :param model: 
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param X_test_noisy: 
    :param X_test_adv: 
    :return: artifacts: positive and negative examples with bu values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Getting Monte Carlo dropout variance predictions...')
    uncerts_normal = get_mc_predictions(model, X_test,
                                        batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)
    uncerts_noisy = get_mc_predictions(model, X_test_noisy,
                                       batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)
    uncerts_adv = get_mc_predictions(model, X_test_adv,
                                     batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)

    print("uncerts_normal:", uncerts_normal.shape)
    print("uncerts_noisy:", uncerts_noisy.shape)
    print("uncerts_adv:", uncerts_adv.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z = normalize(
    #     uncerts_normal,
    #     uncerts_adv,
    #     uncerts_noisy
    # )

    uncerts_pos = uncerts_adv
    uncerts_neg = np.concatenate((uncerts_normal, uncerts_noisy))
    artifacts, labels = merge_and_generate_labels(uncerts_pos, uncerts_neg)

    return artifacts, labels

def get_lid(model, X_test, X_test_noisy, X_test_adv, k=10, batch_size=100, dataset='mnist'):
    """
    Get local intrinsic dimensionality
    :param model: 
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param X_test_noisy: 
    :param X_test_adv: 
    :return: artifacts: positive and negative examples with lid values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Extract local intrinsic dimensionality: k = %s' % k)
    lids_normal, lids_noisy, lids_adv = get_lids_random_batch(model, X_test, X_test_noisy,
                                                              X_test_adv, dataset, k, batch_size)
    print("lids_normal:", lids_normal.shape)
    print("lids_noisy:", lids_noisy.shape)
    print("lids_adv:", lids_adv.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # lids_normal_z, lids_adv_z, lids_noisy_z = normalize(
    #     lids_normal,
    #     lids_adv,
    #     lids_noisy
    # )

    lids_pos = lids_adv
    lids_neg = np.concatenate((lids_normal, lids_noisy))
    artifacts, labels = merge_and_generate_labels(lids_pos, lids_neg)

    return artifacts, labels

def get_kmeans(model, X_test, X_test_noisy, X_test_adv, k=10, batch_size=100, dataset='mnist'):
    """
    Calculate the average distance to k nearest neighbours as a feature.
    This is used to compare density vs LID. Why density doesn't work?
    :param model: 
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param X_test_noisy: 
    :param X_test_adv: 
    :return: artifacts: positive and negative examples with lid values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Extract k means feature: k = %s' % k)
    kms_normal, kms_noisy, kms_adv = get_kmeans_random_batch(model, X_test, X_test_noisy,
                                                              X_test_adv, dataset, k, batch_size,
                                                             pca=True)
    print("kms_normal:", kms_normal.shape)
    print("kms_noisy:", kms_noisy.shape)
    print("kms_adv:", kms_adv.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # kms_normal_z, kms_noisy_z, kms_adv_z = normalize(
    #     kms_normal,
    #     kms_noisy,
    #     kms_adv
    # )

    kms_pos = kms_adv
    kms_neg = np.concatenate((kms_normal, kms_noisy))
    artifacts, labels = merge_and_generate_labels(kms_pos, kms_neg)

    return artifacts, labels

def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw-l2'"
    assert args.artifact in ['kd', 'bu', 'lid', 'km', 'all'], \
        "Artifact(s) to use 'kd', 'bu', 'lid', 'km', 'all'"
    model_file = os.path.join(PATH_DATA, "model_%s.h5" % args.dataset)
    assert os.path.isfile(model_file), \
        'model file not found... must first train model using train_model.py.'
    adv_file = os.path.join(PATH_DATA, "Adv_%s_%s.npy" % (args.dataset, args.attack))
    assert os.path.isfile(adv_file), \
        'adversarial sample file not found... must first craft adversarial ' \
        'samples using craft_adv_samples.py'

    print('Loading the data and model...')
    # Load the model
    model = load_model(model_file)
    # Load the dataset
    X_train, Y_train, X_test, Y_test = get_data(args.dataset)
    # Check attack type, select adversarial and noisy samples accordingly
    print('Loading noisy and adversarial samples...')
    if args.attack == 'all':
        # TODO: implement 'all' option
        # X_test_adv = ...
        # X_test_noisy = ...
        raise NotImplementedError("'All' types detector not yet implemented.")
    else:
        # Load adversarial samples
        X_test_adv = np.load(adv_file)
        print("X_test_adv: ", X_test_adv.shape)

        # as there are some parameters to tune for noisy example, so put the generation
        # step here instead of the adversarial step which can take many hours
        noisy_file = os.path.join(PATH_DATA, 'Noisy_%s_%s.npy' % (args.dataset, args.attack))
        if os.path.isfile(noisy_file):
            X_test_noisy = np.load(noisy_file)
        else:
            # Craft an equal number of noisy samples
            print('Crafting %s noisy samples. ' % args.dataset)
            X_test_noisy = get_noisy_samples(X_test, X_test_adv, args.dataset, args.attack)
            np.save(noisy_file, X_test_noisy)

    # Check model accuracies on each sample type
    for s_type, dataset in zip(['normal', 'noisy', 'adversarial'],
                               [X_test, X_test_noisy, X_test_adv]):
        _, acc = model.evaluate(dataset, Y_test, batch_size=args.batch_size,
                                verbose=0)
        print("Model accuracy on the %s test set: %0.2f%%" %
              (s_type, 100 * acc))
        # Compute and display average perturbation sizes
        if not s_type == 'normal':
            l2_diff = np.linalg.norm(
                dataset.reshape((len(X_test), -1)) -
                X_test.reshape((len(X_test), -1)),
                axis=1
            ).mean()
            print("Average L-2 perturbation size of the %s test set: %0.2f" %
                  (s_type, l2_diff))

    # Refine the normal, noisy and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    preds_test = model.predict_classes(X_test, verbose=0,
                                       batch_size=args.batch_size)
    inds_correct = np.where(preds_test == Y_test.argmax(axis=1))[0]
    print("Number of correctly predict images: %s" % (len(inds_correct)))

    X_test = X_test[inds_correct]
    X_test_noisy = X_test_noisy[inds_correct]
    X_test_adv = X_test_adv[inds_correct]
    print("X_test: ", X_test.shape)
    print("X_test_noisy: ", X_test_noisy.shape)
    print("X_test_adv: ", X_test_adv.shape)

    if args.artifact == 'kd':
        # extract kernel density
        artifacts, labels = get_kd(model, X_train, Y_train, X_test, X_test_noisy, X_test_adv)
        print("KD: [artifact shape: ", artifacts.shape, ", label shape: ", labels.shape)

        # save to file
        bandwidth = BANDWIDTHS[args.dataset]
        file_name = os.path.join(PATH_DATA, 'kd_%s_%s_%.4f.npy' % (args.dataset, args.attack, bandwidth))
        data = np.concatenate((artifacts, labels), axis=1)
        np.save(file_name, data)
    elif args.artifact == 'bu':
        # extract Bayesian uncertainty
        artifacts, labels = get_bu(model, X_test, X_test_noisy, X_test_adv)
        print("BU: [artifact shape: ", artifacts.shape, ", label shape: ", labels.shape)

        # save to file
        file_name = os.path.join(PATH_DATA, 'bu_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((artifacts, labels), axis=1)
        np.save(file_name, data)
    elif args.artifact == 'lid':
        # extract local intrinsic dimensionality
        np.save("/tmp/xtest", X_test)
        np.save("/tmp/xtest_noisy", X_test_noisy)
        np.save("/tmp/xtest_adv", X_test_adv)
        print(args)
        artifacts, labels = get_lid(model, X_test, X_test_noisy, X_test_adv,
                                    args.k_nearest, args.batch_size, args.dataset)
        print("LID: [artifact shape: ", artifacts.shape, ", label shape: ", labels.shape)
        exit(0)
        # save to file
        # file_name = os.path.join(PATH_DATA, 'lid_%s_%s.npy' % (args.dataset, args.attack))
        file_name = os.path.join('../data_grid_search/lid_large_batch/', 'lid_%s_%s_%s.npy' %
                                 (args.dataset, args.attack, args.k_nearest))

        data = np.concatenate((artifacts, labels), axis=1)
        np.save(file_name, data)
    elif args.artifact == 'km':
        # extract k means distance
        artifacts, labels = get_kmeans(model, X_test, X_test_noisy, X_test_adv,
                                    args.k_nearest, args.batch_size, args.dataset)
        print("K-Mean: [artifact shape: ", artifacts.shape, ", label shape: ", labels.shape)

        # save to file
        file_name = os.path.join(PATH_DATA, 'km_pca_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((artifacts, labels), axis=1)
        np.save(file_name, data)
    elif args.artifact == 'all':
        # extract kernel density
        artifacts, labels = get_kd(model, X_train, Y_train, X_test, X_test_noisy, X_test_adv)
        file_name = os.path.join(PATH_DATA, 'kd_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((artifacts, labels), axis=1)
        np.save(file_name, data)

        # extract Bayesian uncertainty
        artifacts, labels = get_bu(model, X_test, X_test_noisy, X_test_adv)
        file_name = os.path.join(PATH_DATA, 'bu_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((artifacts, labels), axis=1)
        np.save(file_name, data)

        # extract local intrinsic dimensionality
        artifacts, labels = get_lid(model, X_test, X_test_noisy, X_test_adv,
                                    args.k_nearest, args.batch_size, args.dataset)
        file_name = os.path.join(PATH_DATA, 'lid_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((artifacts, labels), axis=1)
        np.save(file_name, data)

        # extract k means distance
        # artifacts, labels = get_kmeans(model, X_test, X_test_noisy, X_test_adv,
        #                                args.k_nearest, args.batch_size, args.dataset)
        # file_name = os.path.join(PATH_DATA, 'km_%s_%s.npy' % (args.dataset, args.attack))
        # data = np.concatenate((artifacts, labels), axis=1)
        # np.save(file_name, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'jsma', 'bim-b', 'jsma', 'cw-l2' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-r', '--artifact',
        help="Artifact(s) to use 'kd', 'bu', 'lid' 'km' or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-k', '--k_nearest',
        help="The number of nearest neighbours to use; either 10, 20, 100 ",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=100)
    parser.set_defaults(k_nearest=20)
    # args = parser.parse_args()
    # main(args)

    # parameter-tuning: generation all artifacts if not exist
    for dataset in ['cifar']:
        for attack in ['cw-l2']:
            for k in [20, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
                print('dataset: ', dataset, ', attack: ', attack, ', k_nearest: ', k)
                file_name = os.path.join("../data_grid_search/lid_large_batch/", 'lid_%s_%s_%s.npy' % (dataset, attack, k))
                if os.path.isfile(file_name) and False:
                    print("--skip: lid_%s_%s_%s.npy is already generated!" % (dataset, attack, k))
                    continue
                else:
                    args = parser.parse_args(['-d', dataset, '-a', attack, '-r', 'lid', '-k', str(k), '-b', '1000'])
                    main(args)

    # for dataset in ['cifar']:
    #     for attack in ['cw-l2']:
    #         file_name = os.path.join(PATH_DATA, 'km_pca_%s_%s.npy' % (dataset, attack))
    #         if os.path.isfile(file_name):
    #             print("--skip: km_%s_%s.npy is already generated!" % (dataset, attack))
    #             continue
    #         else:
    #             args = parser.parse_args(['-d', dataset, '-a', attack, '-r', 'km'])
    #             main(args)
