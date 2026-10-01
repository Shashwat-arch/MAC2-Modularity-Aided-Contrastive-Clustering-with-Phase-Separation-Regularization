import numpy as np
from sklearn import metrics
from munkres import Munkres


class clustering_metrics:
    def __init__(self, true_label, predict_label):
        self.true_label = np.array(true_label)
        self.pred_label = np.array(predict_label)

    def clusteringAcc(self):
        numclass = int(self.true_label.max()) + 1
        cost = np.zeros((numclass, numclass), dtype=int)
        for i in range(numclass):
            mps = np.where(self.true_label == i)[0]
            for j in range(numclass):
                cost[i][j] = int(np.sum(self.pred_label[mps] == j))
        indexes = Munkres().compute((-cost).tolist())
        new_pred = np.zeros_like(self.pred_label)
        for i, j in indexes:
            new_pred[self.pred_label == j] = i
        acc = metrics.accuracy_score(self.true_label, new_pred)
        f1_macro = metrics.f1_score(self.true_label, new_pred, average='macro', zero_division=0)
        precision_macro = metrics.precision_score(self.true_label, new_pred, average='macro', zero_division=0)
        recall_macro = metrics.recall_score(self.true_label, new_pred, average='macro', zero_division=0)
        f1_micro = metrics.f1_score(self.true_label, new_pred, average='micro', zero_division=0)
        precision_micro = metrics.precision_score(self.true_label, new_pred, average='micro', zero_division=0)
        recall_micro = metrics.recall_score(self.true_label, new_pred, average='micro', zero_division=0)
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def ARI(self):
        return metrics.adjusted_rand_score(self.true_label, self.pred_label)

    def NMI(self):
        return metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
