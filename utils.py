from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import torch.nn as nn
from sklearn.metrics import roc_auc_score,average_precision_score
class load_data():
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self,data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)

        return epr.T


    def exp_data(self):
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature


def normalize(expression):
    std = StandardScaler()
    epr = std.fit_transform(expression)

    return epr

# def adj2saprse_tensor(adj):
#     coo = adj.tocoo()
#     i = torch.LongTensor([coo.row, coo.col])
#     v = torch.from_numpy(coo.data).float()
#
#     adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
#     return adj_sp_tensor


def Evaluation(y_true, y_pred,flag=False):
    if flag:
        y_p = y_pred[:,-1]
        y_p = y_p.numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.numpy()
        y_p = y_p.flatten()


    y_t = y_true.astype(int)

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)


    AUPR = average_precision_score(y_true=y_t,y_score=y_p)
    AUPR_norm = AUPR/np.mean(y_t)


    return AUC, AUPR, AUPR_norm
def masked_accuracy(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds-labels)
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    error *= mask
#     return tf.reduce_sum(error)
    return tf.sqrt(tf.reduce_mean(error))


def ROC(outs, labels, test_arr, label_neg):
    scores = []
    for i in range(len(test_arr)):
        l = test_arr[i]
        scores.append(outs[int(labels[l, 0] - 1), int(labels[l, 1] - 1)])

    for i in range(label_neg.shape[0]):
        scores.append(outs[int(label_neg[i, 0]), int(label_neg[i, 1])])

    test_labels = np.ones((len(test_arr), 1))
    temp = np.zeros((label_neg.shape[0], 1))
    test_labels1 = np.vstack((test_labels, temp))
    test_labels1 = np.array(test_labels1, dtype=np.bool).reshape([-1, 1])
    return test_labels1, scores
