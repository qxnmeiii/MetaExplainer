
import numpy as np
import torch
import torch.utils.data as utils
from sklearn import preprocessing
import pandas as pd
from scipy.io import loadmat
import pathlib
from sklearn.model_selection import StratifiedKFold
from itertools import compress
import numpy
import h5py
class MaskableList(list):
    def __getitem__(self, index):
        try:
            return super(MaskableList, self).__getitem__(index)
        except TypeError:
            return MaskableList(compress(self, index))
class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def infer_dataloader(dataset_config):

    label_df = pd.read_csv(dataset_config["label"])


    if dataset_config["dataset"] == "PNC":
        fc_data = np.load(dataset_config["time_seires"], allow_pickle=True).item()
        fc_timeseires = fc_data['data'].transpose((0, 2, 1))

        fc_id = fc_data['id']


        id2gender = dict(zip(label_df['SUBJID'], label_df['sex']))

        final_fc, final_label = [], []

        for fc, l in zip(fc_timeseires, fc_id):
            if l in id2gender:
                final_fc.append(fc)
                final_label.append(id2gender[l])
        final_fc = np.array(final_fc)


    elif dataset_config["dataset"] == 'ABCD':

        fc_data = np.load(dataset_config["time_seires"], allow_pickle=True)

    _, node_size, timeseries = final_fc.shape

    encoder = preprocessing.LabelEncoder()

    encoder.fit(label_df["sex"])

    labels = encoder.transform(final_label)

    final_fc = torch.from_numpy(final_fc).float()

    return final_fc, labels, node_size, timeseries


        
def init_dataloader(dataset_config):

    if dataset_config["dataset"] == 'ABIDE':

        data = np.load(dataset_config["time_seires"], allow_pickle=True).item()
        final_fc = data["timeseires"]
        final_pearson = data["corr"]
        labels = data["label"]
        site=data["site"]



    _, _, timeseries = final_fc.shape

    _, node_size, node_feature_size = final_pearson.shape

    scaler = StandardScaler(mean=np.mean(
        final_fc), std=np.std(final_fc))
    
    final_fc = scaler.transform(final_fc)



    final_fc, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_fc, final_pearson, labels)]

    length = final_fc.shape[0]
    train_length = int(length*dataset_config["train_set"])
    val_length = int(length*dataset_config["val_set"])

    dataset = utils.TensorDataset(
        final_fc,
        final_pearson,
        labels
    )



    #torch.manual_seed(111)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_length, val_length, length-train_length-val_length],torch.manual_seed(0))

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    return (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries

def unfold_label(labels, classes):
    new_labels = []

    assert len(np.unique(labels)) == classes
    # minimum value of labels
    mini = np.min(labels)
    mini=int(mini)

    for index in range(len(labels)):
        dump = np.full(shape=[classes], fill_value=0).astype(np.int8)
        _class = int(labels[index]) - mini
        dump[_class] = 1
        new_labels.append(dump)

    return np.array(new_labels)

def shuffle_data(sample1, sample2,labels):
    num = len(labels)
    shuffle_index = np.random.permutation(np.arange(num))
    shuffled_sample1 = sample1[shuffle_index]
    shuffled_sample2 = sample2[shuffle_index]
    shuffled_labels = labels[shuffle_index]
    return shuffled_sample1, shuffled_sample2,shuffled_labels

class BatchImageGenerator:
    def __init__(self,  stage, file_path, b_unfold_label):

        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')

        self.configuration( stage, file_path)
        self.load_data(b_unfold_label)

    def configuration(self,  stage, file_path):
        self.batch_size = 10
        self.current_index = -1
        self.file_path = file_path
        self.stage = stage
        self.shuffled = False
        self.pos_current_index = -1
        self.neg_current_index= -1

    def load_data(self, b_unfold_label):
        file_path = self.file_path
        f = h5py.File(file_path, "r")
        self.final_fc = np.array(f['final_fc'])
        self.final_pearson = np.array(f['final_pearson'])
        self.labels = np.array(f['labels'])
        f.close()
        if b_unfold_label:
            self.labels = unfold_label(labels=self.labels, classes=len(np.unique(self.labels)))
        assert len(self.final_fc) == len(self.labels)

        self.file_num_train = len(self.labels)
        #print('data num loaded:', self.file_num_train)

        if self.stage == 'train':
            self.final_fc,self.final_pearson, self.labels = shuffle_data(sample1=self.final_fc,sample2=self.final_pearson ,labels=self.labels)

    def get_images_labels_batch(self):

        final_fc = []
        final_pearson= []
        labels = []
        for index in range(self.batch_size):
            self.current_index += 1

            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.current_index %= self.file_num_train

                self.final_fc,self.final_pearson, self.labels = shuffle_data(sample1=self.final_fc,sample2=self.final_pearson ,labels=self.labels)

            final_fc.append(self.final_fc[self.current_index])
            final_pearson.append(self.final_pearson[self.current_index])
            labels.append(self.labels[self.current_index])

        final_fc = np.stack(final_fc)
        final_pearson = np.stack(final_pearson)
        labels = np.stack(labels)

        return final_fc,final_pearson, labels

    def get_images_labels_batch_new(self):

        final_fc = []
        final_pearson= []
        labels = []
        pos_indexes = np.where(self.labels == 0)[0]
        neg_indexes=np.where(self.labels == 1)[0]
        for index in range(self.batch_size//2):
            self.pos_current_index += 1
            self.neg_current_index += 1
            if self.pos_current_index > len(pos_indexes)-1 or self.neg_current_index > len(neg_indexes) - 1:
                # if self.pos_current_index > len(pos_indexes) - 1:
                #     self.pos_current_index %= len(pos_indexes)
                # if self.pos_current_index > len(pos_indexes) - 1:
                #     self.pos_current_index %= len(pos_indexes)
                self.pos_current_index = 0
                self.neg_current_index = 0

                self.final_fc, self.final_pearson, self.labels = shuffle_data(sample1=self.final_fc,
                                                                              sample2=self.final_pearson,
                                                                              labels=self.labels)
                pos_indexes = np.where(self.labels == 0)[0]
                neg_indexes = np.where(self.labels == 1)[0]

            final_fc.append(self.final_fc[pos_indexes[self.pos_current_index]])
            final_pearson.append(self.final_pearson[pos_indexes[self.pos_current_index]])
            labels.append(self.labels[pos_indexes[self.pos_current_index]])

            final_fc.append(self.final_fc[neg_indexes[self.neg_current_index]])
            final_pearson.append(self.final_pearson[neg_indexes[self.neg_current_index]])
            labels.append(self.labels[neg_indexes[self.neg_current_index]])

        final_fc = np.stack(final_fc)
        final_pearson = np.stack(final_pearson)
        labels = np.stack(labels)
        final_fc, final_pearson, labels = shuffle_data(sample1=final_fc,sample2=final_pearson,labels=labels)

        return final_fc,final_pearson, labels


def init_dataloader1(dataset_config):
    if dataset_config["dataset"] == 'ABIDE':
        f1 = h5py.File("braincnntrain.h5", "r")
        final_fc1 = f1['final_fc']
        labels1 = f1['labels']
        final_pearson1 = f1['final_pearson']


    _, _, timeseries1 = final_fc1.shape

    _, node_size1, node_feature_size1 = final_pearson1.shape

    scaler = StandardScaler(mean=np.mean(
        final_fc1), std=np.std(final_fc1))

    final_fc1 = scaler.transform(final_fc1)

    final_pearson1=np.array(final_pearson1)
    labels1= np.array(labels1)



    final_fc1, final_pearson1, labels1 = [torch.from_numpy(
        data).float() for data in (final_fc1, final_pearson1, labels1)]

    length1 = final_fc1.shape[0]
    train_length = int(length1 * 0.8)
    #val_length = int(length1 * 0.2)

    dataset1 = utils.TensorDataset(
        final_fc1,
        final_pearson1,
        labels1
    )

    # torch.manual_seed(111)

    train_dataset, val_dataset= torch.utils.data.random_split(
        dataset1, [train_length, length1-train_length], torch.manual_seed(0))

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    f2 = h5py.File("braincnntest.h5", "r")
    final_fc2 = f2['final_fc']
    labels2 = f2['labels']
    final_pearson2 = f2['final_pearson']
    scaler = StandardScaler(mean=np.mean(
        final_fc2), std=np.std(final_fc2))

    final_fc2 = scaler.transform(final_fc2)
    final_pearson2 = np.array(final_pearson2)
    labels2 = np.array(labels2)

    final_fc2, final_pearson2, labels2 = [torch.from_numpy(
        data).float() for data in (final_fc2, final_pearson2, labels2)]
    dataset2 = utils.TensorDataset(
        final_fc2,
        final_pearson2,
        labels2
    )

    test_dataloader = utils.DataLoader(
        dataset2, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    return (train_dataloader, val_dataloader, test_dataloader), node_size1, node_feature_size1, timeseries1
