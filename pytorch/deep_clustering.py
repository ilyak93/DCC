from os import path

import sklearn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from config import cfg
from easydict import EasyDict as edict
from edgeConstruction import compressed_data
import matplotlib.pyplot as plt
import data_params as dp
import pretraining
import extract_feature
import copyGraph
import DCC
import scipy.io as sio
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import pathlib
import numpy as np


#cur_dir = '/content/gdrive/My Drive/ssh_files/submitted - Copy'
cur_dir = str(pathlib.Path().absolute())
k = 10

compressed_data(cur_dir, k, preprocess='none', algo='mknn', isPCA=None, format='mat')


args = edict()
args.db = 'lstm'
args.niter = 5000
args.step = 2000
args.lr = 0.01
args.resume = True
args.level = 0
args.batchsize = 256
args.ngpu = 4
args.deviceID = 0
args.tensorboard = True
args.h5 = False
args.id = 10
args.dim = 2049 #frame length
args.manualSeed = cfg.RNG_SEED
args.clean_log = True
args.data_path = str(cur_dir)
args.out_path = str(cur_dir)

index, net = pretraining.main(args)
args.feat = 'pretrained'
args.torchmodel = 'checkpoint_{}.pth.tar'.format(index)
extract_feature.main(args, net=net)


args.g = 'pretrained.mat'
args.out = 'pretrained'
args.feat = 'pretrained.pkl'
copyGraph.main(args)

args.batchsize = cfg.PAIRS_PER_BATCH
#args.batchsize = 256
args.resume = False
args.level = 180
args.nepoch = 30000
args.M = 20
args.lr = 0.001
out = DCC.main(args, net=net)
print('Done')
exit(0)

# Z=Y initial
cur_dir = str(pathlib.Path().absolute())
out = sio.loadmat(cur_dir+'/features')

Z = out['Z']
labels = out['gtlabels'].squeeze()

'''
feat_cols = [str(i) for i in range(Z.shape[1])]
df = pd.DataFrame(Z, columns=feat_cols)
df['y'] = labels
df['label'] = df['y'].apply(lambda i: str(i))

tsne = TSNE(n_components=2, verbose=1, perplexity=30)
tsne_results = tsne.fit_transform(df[feat_cols].values)

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df,
    legend="full",
    alpha=0.3
)
plt.show()

'''
'''
import hdbscan
cluster_dict = {}
clusterer = hdbscan.HDBSCAN(min_cluster_size=6, gen_min_span_tree=True)
hdbscan = clusterer.fit_predict(Z)

for idx, cluster in enumerate(hdbscan):
    gt_label = labels[idx]
    if not(cluster in cluster_dict.keys()):
        cluster_dict[cluster] = {1 : 0, 2 : 0, 4 : 0}
    cluster_dict[cluster][gt_label] += 1
cluster_dict1 = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: item[1][1], reverse=True)}
cluster_dict2 = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: item[1][2], reverse=True)}
cluster_dict4 = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: item[1][4], reverse=True)}
print(cluster_dict)
'''


# tuning hyperparmeters function
def dbscan_hyperparameters_tuning(df, eps_grid, min_samples_grid, dist_metric):


    labels_dict_lst = []

    eps_lst = []

    min_samples_lst = []

    silhouette_scores_lst = []

    calinski_harabasz_scores_lst = []

    davies_bouldin_score_lst = []

    for eps in eps_grid:
        for min_samples in min_samples_grid:
            model = DBSCAN(eps=eps, min_samples=min_samples, metric=dist_metric, n_jobs=-1)
            labels = model.fit_predict(df)
            unique, counts = np.unique(labels, return_counts=True)
            labels_dict = dict(zip(unique, counts))
            labels_dict_lst.append(labels_dict)
            eps_lst.append(eps)
            min_samples_lst.append(min_samples)
            try:
                silh_score = silhouette_score(df, labels, metric='euclidean')
                ch_score = calinski_harabasz_score(df, labels)
                db_score = davies_bouldin_score(df, labels)
            except:
                silh_score = 0
                ch_score = 0
                db_score = 0
                silhouette_scores_lst.append(silh_score)
                calinski_harabasz_scores_lst.append(ch_score)
                davies_bouldin_score_lst.append(db_score)
                continue
            silhouette_scores_lst.append(silh_score)
            calinski_harabasz_scores_lst.append(ch_score)
            davies_bouldin_score_lst.append(db_score)
    labels_dict_lst = [{k: v for k, v in sorted(dct.items(), key=lambda item: item[1], reverse=True)} for dct in labels_dict_lst]
    results = pd.DataFrame({'eps': eps_lst,
                            'min_samples': min_samples_lst,
                            'silhouette': silhouette_scores_lst,
                            'calinski_harabasz': calinski_harabasz_scores_lst,
                            'davies_bouldin': davies_bouldin_score_lst,
                            'labels': labels_dict_lst})

    return results

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(Z)
distances, indices = nbrs.kneighbors(Z)

distances = np.sort(distances, axis=0)
plt.plot(distances)
plt.show()


eps_values = [0.01, 0.02, 0.03, 0.04,  0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35]

min_samples_values = [1, 2, 3, 4, 5, 10, 20, 30, 50, 100]

results = dbscan_hyperparameters_tuning(Z, eps_values, min_samples_values, 'euclidean')
results.to_csv(cur_dir+'/hp.csv')

'''
dbscan = DBSCAN(eps=0.4,  min_samples=100).fit_predict(Z)
cluster_dict = {}
for idx, cluster in enumerate(dbscan):
    gt_label = labels[idx]
    if not(cluster in cluster_dict.keys()):
        cluster_dict[cluster] = {1 : 0, 2 : 0, 4 : 0}
    cluster_dict[cluster][gt_label] += 1
cluster_dict1 = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: item[1][1], reverse=True)}
cluster_dict2 = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: item[1][2], reverse=True)}
cluster_dict4 = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: item[1][4], reverse=True)}
print(cluster_dict)

cluster_dict = {}
kmeans = KMeans(n_clusters=5, random_state=0).fit(Z)

for idx, cluster in enumerate(kmeans.labels_):
    gt_label = labels[idx]
    if not(cluster in cluster_dict.keys()):
        cluster_dict[cluster] = {1 : 0, 2 : 0, 4 : 0}
    cluster_dict[cluster][gt_label] += 1
cluster_dict = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: sum(item[1].values()), reverse=True)}
print(cluster_dict)




final = str(pathlib.Path('features').absolute())
if path.exists(final):
    out = sio.loadmat(final)

    #final output
    embedded = out['U']

    y = out['gtlabels'].squeeze()

    feat_cols = [str(i) for i in range(embedded.shape[1])]
    df = pd.DataFrame(embedded, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    tsne = TSNE(n_components=2, verbose=1, perplexity=30)
    tsne_results = tsne.fit_transform(df[feat_cols].values)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]


    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 3),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()

    cluster_dict = {}
    clusters = np.unique(out['cluster'].squeeze())
    print('number of clusters'+str(clusters))

    for idx, cluster in enumerate(out['cluster'].squeeze()):

        gt_label = y[idx]

        if not(cluster in cluster_dict.keys()):
            cluster_dict[cluster] = {1 : 0, 2 : 0, 4 : 0}

        cluster_dict[cluster][gt_label] += 1

    cluster_dict = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: sum(item[1].values()), reverse=True)}
    print(cluster_dict)

    
    cluster_dict = {}
    kmeans = KMeans(n_clusters=5, random_state=0).fit(embedded)

    for idx, cluster in enumerate(kmeans.labels_):
        gt_label = labels[idx]
        if not(cluster in cluster_dict.keys()):
            cluster_dict[cluster] = {1 : 0, 2 : 0, 4 : 0}
        cluster_dict[cluster][gt_label] += 1
    cluster_dict = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: sum(item[1].values()), reverse=True)}
    print(cluster_dict)



    #final Y output
    embedded = out['Z']

    y = out['gtlabels'].squeeze()

    feat_cols = [str(i) for i in range(embedded.shape[1])]
    df = pd.DataFrame(embedded, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    tsne = TSNE(n_components=2, verbose=1, perplexity=30)
    tsne_results = tsne.fit_transform(df[feat_cols].values)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]


    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 3),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()

    cluster_dict = {}
    kmeans = KMeans(n_clusters=5, random_state=0).fit(embedded)

    for idx, cluster in enumerate(kmeans.labels_):
        gt_label = labels[idx]
        if not(cluster in cluster_dict.keys()):
            cluster_dict[cluster] = {1 : 0, 2 : 0, 4 : 0}
        cluster_dict[cluster][gt_label] += 1
    cluster_dict = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: sum(item[1].values()), reverse=True)}
    print(cluster_dict)



#mid results
for i in range (0, 501, 500):
    out = sio.loadmat(cur_dir+'/features'+str(i))
    #Z output
    embedded = out['U']

    y = out['gtlabels'].squeeze()

    feat_cols = [str(i) for i in range(embedded.shape[1])]
    df = pd.DataFrame(embedded, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    tsne = TSNE(n_components=2, verbose=1, perplexity=30)
    tsne_results = tsne.fit_transform(df[feat_cols].values)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 3),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()

    cluster_dict = {}
    clusters = np.unique(out['cluster'].squeeze())
    print('number of clusters' + str(clusters))

    for idx, cluster in enumerate(out['cluster'].squeeze()):

        gt_label = y[idx]

        if not (cluster in cluster_dict.keys()):
            cluster_dict[cluster] = {1: 0, 2: 0, 4: 0}

        cluster_dict[cluster][gt_label] += 1

    cluster_dict = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: sum(item[1].values()), reverse=True)}
    print(cluster_dict)

    cluster_dict = {}
    kmeans = KMeans(n_clusters=5, random_state=0).fit(embedded)

    for idx, cluster in enumerate(kmeans.labels_):
        gt_label = labels[idx]
        if not(cluster in cluster_dict.keys()):
            cluster_dict[cluster] = {1 : 0, 2 : 0, 4 : 0}
        cluster_dict[cluster][gt_label] += 1
    cluster_dict = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: sum(item[1].values()), reverse=True)}
    print(cluster_dict)

    # Y output
    embedded = out['Z']

    y = out['gtlabels'].squeeze()

    feat_cols = [str(i) for i in range(embedded.shape[1])]
    df = pd.DataFrame(embedded, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    tsne = TSNE(n_components=2, verbose=1, perplexity=30)
    tsne_results = tsne.fit_transform(df[feat_cols].values)

    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 3),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()

    cluster_dict = {}
    kmeans = KMeans(n_clusters=5, random_state=0).fit(embedded)

    for idx, cluster in enumerate(kmeans.labels_):
        gt_label = labels[idx]
        if not (cluster in cluster_dict.keys()):
            cluster_dict[cluster] = {1: 0, 2: 0, 4: 0}
        cluster_dict[cluster][gt_label] += 1
    cluster_dict = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: sum(item[1].values()), reverse=True)}
    print(cluster_dict)



    df['y'] = out['cluster'].squeeze()
    colors = np.unique(out['cluster'].squeeze()).size
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", colors),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()
'''