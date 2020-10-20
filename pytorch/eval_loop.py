import argparse
import pathlib

import numpy as np
import matplotlib.pylab as plt
import sys
import pandas as pd
import sklearn, sklearn.model_selection, sklearn.neighbors
import sklearn.linear_model
from mlp import MLP, MLP_train
import gzip
import utils
import collections
from tqdm import tqdm
import scipy.io as sio

from easydict import EasyDict as edict

'''
parser = argparse.ArgumentParser()
parser.add_argument('embeddings_file', help='File with embeddings')
parser.add_argument('-num_examples', nargs='?', type=int, default=20000, help='')
parser.add_argument('-num_trials', nargs='?', type=int, default=10, help='')
parser.add_argument('-labels_file', nargs='?', default="test_labels.csv.gz", help='')
parser.add_argument('-model', type=str, default="knn", choices=["knn", "mlp", "lr", "adaboost"],
                    help='Model to evaluate embeddings with.')
args = parser.parse_args()

print(args)
'''

## get counts
# lines_emb = 0
# with gzip.open(args.embeddings_file, 'rb') as f:
#    for line in f:
#        lines_emb += 1

# lines_labels = 0
# with gzip.open(args.labels_file, 'rb') as f:
#     for line in f:
#         lines_labels += 1

# print("lines_emb:", lines_emb)

# if lines_labels != lines_emb:
#     print(" !! Issue with coverage of labels. The data must align to the labels.")
#     sys.exit()
if __name__ == "__main__":
    args = edict()
    cur_dir = pathlib.Path().absolute()
    out = sio.loadmat(cur_dir/'features')
    args.num_examples = out['U'].shape[0]
    args.num_trials=10

    args.X = out['U']
    args.labels =  out['gtlabels'].T

    args.model = 'knn'
    embedded = out['U']

    y = out['gtlabels'].squeeze()

    feat_cols = [str(i) for i in range(embedded.shape[1])]
    df = pd.DataFrame(embedded, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=30)
    tsne_results = tsne.fit_transform(df[feat_cols].values)

    new_df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])
    new_df['tsne-2d-one'] = tsne_results[:, 0]
    new_df['tsne-2d-two'] = tsne_results[:, 1]

    f = open("knn_results_None_" + str(args.num_examples) + ".txt", 'w')
    for Lr in [0.01]:
        for drop_rate in [0.1]:

            def evaluate(num_examples, num_trials):
                all_acc = []
                for i in range(num_trials):

                    print("Generating subset", i)

                    data, labels = new_df, args.labels

                    X, X_test, y, y_test = \
                        sklearn.model_selection.train_test_split(data, labels,
                                                                 train_size=int(0.95*num_examples),
                                                                 stratify=labels,
                                                                 random_state=i)
                    # print("X", X.shape, "X_test", X_test.shape)
                    if args.model == "knn":
                        model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10)
                    elif args.model == "lr":
                        model = sklearn.linear_model.LogisticRegression(multi_class="auto")
                    elif args.model == "adaboost":
                        model = sklearn.ensemble.AdaBoostClassifier()
                    elif args.model == "mlp":
                        num_labels = len(np.unique(y))
                        input_size = X.shape[1]
                        layers = [input_size, int(input_size / 1.5), int(input_size / 1.5), int(input_size / 2),
                                  int(input_size / 2)]
                        network = MLP(input_size, layers, num_labels, dropout_rate=drop_rate)
                        model = MLP_train(network, lr=Lr)
                    else:
                        print("Unknown model")
                        sys.exit()

                    # print(model)
                    model = model.fit(X, y.flatten())
                    y_pred = model.predict(X_test)
                    bacc = sklearn.metrics.balanced_accuracy_score(y_test.flatten(), y_pred)
                    all_acc.append(bacc)
                    print("   Run {}".format(i) + ", label_type: {}".format('btype') + ", Balanced Accuracy: {}".format(bacc))

                return np.asarray(all_acc).mean(), np.asarray(all_acc).std()


            btype_mean, btype_stdev = evaluate(args.num_examples, args.num_trials)
            #rtype_mean, rtype_stdev = evaluate(args.num_examples, args.num_trials, "rtype")

            print("btype, Accuracy:", round(btype_mean, 3), "+-", round(btype_stdev, 3), "drop:", drop_rate, "lr:", Lr,
                  args)
            #print("rtype,Accuracy:", round(rtype_mean, 3), "+-", round(rtype_stdev, 3), "drop:", drop_rate, "lr:", Lr, args,
                  #file=f)