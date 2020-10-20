import os
import gzip
import pandas as pd
import scipy.io as sio
import pathlib

from enum import Enum

import torch
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler


class Btype(Enum):
    Undefined = 0
    Normal = 1
    ESSV_aka_PAC = 2
    Aberrated = 3
    ESV_aka_PVC = 4

class Rtype(Enum):
    Undefined = 0
    End = 1
    Noise = 2
    NSR = 3
    AFib = 4
    AFlutter = 5


cur_dir = '/content/gdrive/My Drive/ssh_files/submitted - Copy'
emb_file  = cur_dir+'/test_emb.csv.gz'
labels_file = cur_dir+'/test_labels.csv.gz'

train_patients = 1000
test_patients = 100

train_subset = []
test_subset = []

train_labels_subset = []
test_labels_subset = []

#change this param to choose amount of data to work with
partition = 10


with gzip.open(labels_file, 'rb') as f:
    header = f.readline().decode('ascii').replace("\n","").replace("\r", "")
    for i, line in enumerate(f):
        decoded = line.decode('ascii').replace("\n", "").replace("\r", "").split(",")
        patient, segment, frame = decoded[1], decoded[2], decoded[3]
        btype, rtype = Btype(int(decoded[4])), Rtype(int(decoded[5]))
        if btype in [Btype.Undefined] or rtype in [Rtype.Undefined, Rtype.End, Rtype.Noise]:
            continue
		
		# also can add condition on rtype or btype or any other to manipulate the learning data
        if int(patient) < train_patients and int(patient) % partition:
            train_labels_subset.append([patient, segment, frame, btype.value, rtype.value])
        elif int(patient) % partition:
            test_labels_subset.append([patient, segment, frame, btype.value, rtype.value])


train_labels = pd.DataFrame(train_labels_subset, columns=['patient', 'segment', 'frame', 'bt_label', 'rt_label'])
test_labels = pd.DataFrame(test_labels_subset, columns=['patient', 'segment', 'frame', 'bt_label', 'rt_label'])


train_indices = []
test_indices = []

for index, row in train_labels.iterrows():
    train_indices.append(row['patient'] + '_' + row['segment'] + '_' + row['frame'])
for index, row in test_labels.iterrows():
    test_indices.append(row['patient'] + '_' + row['segment'] + '_' + row['frame'])

bt_train_labels = train_labels.drop(['patient', 'segment', 'frame', 'rt_label'], axis=1)
bt_test_labels = test_labels.drop(['patient', 'segment', 'frame', 'rt_label'], axis=1)

rt_train_labels = train_labels.drop(['patient', 'segment', 'frame', 'bt_label'], axis=1)
rt_test_labels = test_labels.drop(['patient', 'segment', 'frame', 'bt_label'], axis=1)


with gzip.open(emb_file, 'rb') as f:
    header = f.readline().decode('ascii').replace("\n", "").replace("\r", "")
    print(header)
    for i, line in enumerate(f):
        row = line.decode('ascii').replace("\n","").replace("\r", "").split(",")
        patient, segment, frame = row[0], row[1], row[2]
        print(patient)
        row = [float(i) for i in row]  # convert here for memory
        frame_data = row[3:]
        if patient + '_' + segment + '_' + frame in train_indices:
            train_subset.append(frame_data)
        elif patient + '_' + segment + '_' + frame in test_indices:
            test_subset.append(frame_data)

train_data = pd.DataFrame(train_subset).astype("float32")
test_data = pd.DataFrame(test_subset).astype("float32")

total_data = pd.concat([train_data, test_data])
total_data = total_data.to_numpy()


import matplotlib.pyplot as plt
import numpy as np

#plotting the heat map of the features and working with different normalization/scaling tools
plt.imshow(total_data, cmap='hot', interpolation='nearest')
plt.savefig('before.pdf', figsize=(18, 18))
#minmaxscale = MinMaxScaler().fit(total_data)
#total_data = minmaxscale.transform(total_data)
#scaler = StandardScaler()
# Fit your data on the scaler object
#total_data = scaler.fit_transform(total_data)
n_samples, n_features = total_data.shape
total_data = np.sqrt(n_features) * normalize(total_data, copy=False)

plt.imshow(total_data, cmap='hot', interpolation='nearest')
plt.savefig('after.pdf', figsize=(18, 18))

train_length = train_data.shape[0]

train_data = total_data[0:train_length]
test_data = total_data[train_length:]


bt_train_labels = bt_train_labels.to_numpy()
bt_test_labels = bt_test_labels.to_numpy()

rt_train_labels = rt_train_labels.to_numpy()
rt_test_labels = rt_test_labels.to_numpy()

# uncomment this if you not using any scaling/normalization
#train_data = train_data.to_numpy()
#test_data = test_data.to_numpy()

print('len='+str(bt_train_labels.shape[0]))


sio.savemat(cur_dir+'/traindata_bt.mat', {'X': train_data, 'Y': bt_train_labels})
sio.savemat(cur_dir+'/testdata_bt.mat', {'X': test_data, 'Y': bt_test_labels})

sio.savemat(cur_dir+'/traindata_rt.mat', {'X': train_data, 'Y': rt_train_labels})
sio.savemat(cur_dir+'/testdata_rt.mat', {'X': test_data, 'Y': rt_test_labels})


