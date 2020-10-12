from time import time
import os
import numpy as np
import scipy.io as sio
import argparse
import random

from config import cfg, get_data_dir, get_output_dir

from sklearn.preprocessing import scale as skscale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import distance_metrics


def load_data(filename, n_samples):
    import cPickle
    fo = open(filename, 'rb')
    data = cPickle.load(fo)
    fo.close()
    labels = data['labels'][0:n_samples]
    labels = np.squeeze(labels)
    features = data['data'][0:n_samples]
    features = features.astype(np.float32, copy=False)
    features = features.reshape((n_samples, -1))
    return labels, features


def load_matdata(filename, n_samples):
    # TODO switch other loading to also use new X,Y convention instead of labels,data?
    data = sio.loadmat(filename)
    labels = data['Y'][0:n_samples]
    labels = np.squeeze(labels)
    features = data['X'][0:n_samples]
    features = features.astype(np.float32, copy=False)
    # TODO figure out why we need to reshape this...
    # features = features.reshape((n_samples, -1))
    return labels, features


def load_data_h5py(filename, n_samples):
    import h5py
    data = h5py.File(filename, 'r')
    labels = data['labels'][0:n_samples]
    labels = np.squeeze(labels)
    features = data['data'][0:n_samples]
    features = features.astype(np.float32, copy=False)
    features = features.reshape((n_samples, -1))
    data.close()
    return labels, features


def load_train_and_validation(loader, datadir, n_samples):
    td = os.path.join(datadir, 'traindata.mat')
    # TODO n_samples don't really make sense as a single parameter anymore since data set is split in 2
    lt, ft = loader(td, n_samples)

    tv = os.path.join(datadir, 'testdata.mat')
    lv, fv = loader(tv, n_samples)

    return np.concatenate((lt, lv)), np.concatenate((ft, fv))


def feature_transformation(features, preprocessing='normalization'):
    n_samples, n_features = features.shape
    if preprocessing == 'scale':
        features = skscale(features, copy=False)
    elif preprocessing == 'minmax':
        minmax_scale = MinMaxScaler().fit(features)
        features = minmax_scale.transform(features)
    elif preprocessing == 'normalization':
        features = np.sqrt(n_features) * normalize(features, copy=False)
    else:
        print('No preprocessing is applied')
    return features


def kNN(X, k, measure='euclidean'):
    """
    Construct pairwise weights by finding the k nearest neighbors to each point
    and assigning a Gaussian-based distance.

    Parameters
    ----------
    X : [n_samples, n_dim] array
    k : int
        number of neighbors for each sample in X
    """
    from scipy.spatial import distance

    weights = []
    parallelized_metrics = list(distance_metrics().keys())
        
    if (measure in parallelized_metrics):
        w = pairwise_distances(X=X, Y=X, metric=measure, n_jobs=-1)
    else:
        w = distance.cdist(X, X, measure)
            
    y = np.argsort(w, axis=1)

    for i, x in enumerate(X):
        distances, indices = w[i, y[i, 1:k + 1]], y[i, 1:k + 1]
        for (d, j) in zip(distances, indices):
            if i < j:
                weights.append((i, j, d * d))
            else:
                weights.append((j, i, d * d))
    weights = sorted(weights, key=lambda r: (r[0], r[1]))
    return np.unique(np.asarray(weights), axis=0)


def mkNN(X, k, measure='euclidean'):
    """
    Construct mutual_kNN for large scale dataset

    If j is one of i's closest neighbors and i is also one of j's closest members,
    the edge will appear once with (i,j) where i < j.

    Parameters
    ----------
    X : [n_samples, n_dim] array
    k : int
      number of neighbors for each sample in X
    """
    from scipy.spatial import distance
    from scipy.sparse import csr_matrix, triu, find
    from scipy.sparse.csgraph import minimum_spanning_tree

    samples = X.shape[0]
    batchsize = 10000
    b = np.arange(k + 1)
    b = tuple(b[1:].ravel())

    z = np.zeros((samples, k))
    weigh = np.zeros_like(z)

    # This loop speeds up the computation by operating in batches
    # This can be parallelized to further utilize CPU/GPU resource
    for x in np.arange(0, samples, batchsize):
        start = x
        end = min(x + batchsize, samples)
        
        parallelized_metrics = list(distance_metrics().keys())
        
        if (measure in parallelized_metrics):
            w = pairwise_distances(X=X[start:end], Y=X, metric=measure, n_jobs=-1)
        else:
            w = distance.cdist(X[start:end], X, measure)

        y = np.argpartition(w, b, axis=1)

        z[start:end, :] = y[:, 1:k + 1]
        weigh[start:end, :] = np.reshape(w[tuple(np.repeat(np.arange(end - start), k)), tuple(y[:, 1:k + 1].ravel())],
                                         (end - start, k))
        del (w)

    ind = np.repeat(np.arange(samples), k)

    P = csr_matrix((np.ones((samples * k)), (ind.ravel(), z.ravel())), shape=(samples, samples))
    Q = csr_matrix((weigh.ravel(), (ind.ravel(), z.ravel())), shape=(samples, samples))

    Tcsr = minimum_spanning_tree(Q)
    P = P.minimum(P.transpose()) + Tcsr.maximum(Tcsr.transpose())
    P = triu(P, k=1)

    return np.asarray(find(P)).T


def compressed_data(dataset, n_samples, k, preprocess=None, algo='mknn', isPCA=None, format='mat'):
    datadir = get_data_dir(dataset)
    if format == 'pkl':
        labels, features = load_train_and_validation(load_data, datadir, n_samples)
    elif format == 'h5':
        labels, features = load_train_and_validation(load_data_h5py, datadir, n_samples)
    else:
        labels, features = load_train_and_validation(load_matdata, datadir, n_samples)

    features = feature_transformation(features, preprocessing=preprocess)

    # PCA is computed for Text dataset. Please refer RCC paper for exact details.
    features1 = features.copy()
    if isPCA is not None:
        pca = PCA(n_components=isPCA, svd_solver='full').fit(features)
        features1 = pca.transform(features)

    t0 = time()

    if algo == 'knn':
        weights = kNN(features1, k=k, measure='euclidean')
    else:
        weights = mkNN(features1, k=k, measure='cosine')

    print('The time taken for edge set computation is {}'.format(time() - t0))

    filepath = os.path.join(datadir, 'pretrained')
    if format == 'h5':
        import h5py
        fo = h5py.File(filepath + '.h5', 'w')
        fo.create_dataset('X', data=features)
        fo.create_dataset('w', data=weights[:, :2])
        fo.create_dataset('gtlabels', data=labels)
        fo.close()
    else:
        sio.savemat(filepath + '.mat', mdict={'X': features, 'w': weights[:, :2], 'gtlabels': labels})


def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Feature extraction for RCC algorithm')

    parser.add_argument('--dataset', default=None, type=str,
                        help='The entered dataset file must be in the Data folder')
    parser.add_argument('--prep', dest='prep', default='none', type=str,
                        help='preprocessing of data: scale,minmax,normalization,none')
    parser.add_argument('--algo', dest='algo', default='mknn', type=str,
                        help='Algorithm to use: knn,mknn')
    parser.add_argument('--k', dest='k', default=10, type=int,
                        help='Number of nearest neighbor to consider')
    parser.add_argument('--pca', dest='pca', default=None, type=int,
                        help='Dimension of PCA processing before kNN graph construction')
    parser.add_argument('--samples', dest='nsamples', default=0, type=int,
                        help='total samples to consider')
    parser.add_argument('--format', choices=['mat', 'pkl', 'h5'], default='mat', help='Dataset format')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
   -----------------------------
   Dataset	|samples| dimension
   -----------------------------
   Mnist	|70000	| [28,28,1]
   YaleB	|2414	| [168,192,1]
   Coil100	|7200	| [128,128,3]
   YTF  	|10056	| [55,55,3]
   Reuters	|9082	| 2000
   RCV1		|10000	| 2000 
   -----------------------------   
   """

    random.seed(50)

    args = parse_args()
    print('Called with args:')
    print(args)

    # storing compressed data
    compressed_data(args.dataset, args.nsamples, args.k, preprocess=args.prep, algo=args.algo, isPCA=args.pca,
                    format=args.format)
