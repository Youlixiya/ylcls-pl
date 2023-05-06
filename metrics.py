import numpy as np
import torch
from sklearn.metrics import confusion_matrix

class Metrics:
    def __init__(self, cls_num):
        self.cls_num = cls_num
        self.cls_index_list = list(range(cls_num))
        self.metrics_dict = dict(
            train_loss=0, train_acc=0,
            train_loss_list=[], valid_loss_list=[],
            valid_loss=0, valid_acc=0,
            train_acc_list=[], valid_acc_list=[],
            valid_best_acc=0,
            valid_best_precision_list=np.zeros(cls_num), valid_best_recall_list=np.zeros(cls_num), valid_best_f1_list=np.zeros(cls_num),
            train_preds_list=[], valid_preds_list=[],
            train_labels_list=[], valid_labels_list=[],
            train_confusion_matrix=None,
            valid_confusion_matrix=None,
            train_precision_list=np.zeros(cls_num), valid_precision_list=np.zeros(cls_num),
            train_recall_list=np.zeros(cls_num), valid_recall_list=np.zeros(cls_num),
            train_f1_list=np.zeros(cls_num), valid_f1_list=np.zeros(cls_num),
            train_total_precision_list=[], valid_total_precision_list=[],
            train_total_recall_list=[], valid_total_recall_list=[],
            train_total_f1_list=[], valid_total_f1_list=[],
            train_mean_precision_list=[], valid_mean_precision_list=[],
            train_mean_recall_list=[], valid_mean_recall_list=[],
            train_mean_f1_list=[], valid_mean_f1_list=[],
        )
        self.is_count_accuracy = False
        self.is_count_precision = False
        self.is_count_recall = False
        self.is_count_f1 = False
        self.is_count_loss = False
    def load_metrics_dict(self, metrics_dict):
        self.metrics_dict = metrics_dict
    def callback_on_per_epoch_start(self, mode='train'):
        self.is_count_accuracy = False
        self.is_count_precision = False
        self.is_count_recall = False
        self.is_count_f1 = False
        self.is_count_loss = False
        self.metrics_dict[f'{mode}_loss']=0
        # self.metrics_dict[f'{mode}_confusion_matrix'] = np.zeros([self.cls_num, self.cls_num]),
        self.metrics_dict[f'{mode}_preds_list'] = []
        self.metrics_dict[f'{mode}_labels_list'] = []
        self.metrics_dict[f'{mode}_precision'] = np.zeros(self.cls_num)
        self.metrics_dict[f'{mode}_recall'] = np.zeros(self.cls_num)
        self.metrics_dict[f'{mode}_f1'] = np.zeros(self.cls_num)
    def callback_on_per_step(self, outputs, labels, loss, mode='train'):
        self.metrics_dict[f'{mode}_loss'] += loss.detach().item()
        preds = outputs.argmax(-1).squeeze()
        self.metrics_dict[f'{mode}_preds_list'].append(preds)
        self.metrics_dict[f'{mode}_labels_list'].append(labels)
        # for i in range(preds.shape[0]):
        #     self.metrics_dict[f'{mode}_confusion_matrix'][labels[i].item(), preds[i].item()] += 1
    def callback_on_per_epoch_end(self, mode = 'train'):
        preds = torch.cat(self.metrics_dict[f'{mode}_preds_list'], dim=0).cpu().numpy()
        labels = torch.cat(self.metrics_dict[f'{mode}_labels_list'], dim=0).cpu().numpy()
        self.metrics_dict[f'{mode}_confusion_matrix'] = confusion_matrix(labels, preds, labels=self.cls_index_list)
        accuracy, precision, recall, f1, loss = self.count_metrics(mode)
        mean_precision = round(precision.mean(), 2)
        mean_recall = round(recall.mean(), 2)
        mean_f1 = round(f1.mean(), 2)
        self.metrics_dict[f'{mode}_mean_precision_list'].append(mean_precision)
        self.metrics_dict[f'{mode}_mean_recall_list'].append(mean_recall)
        self.metrics_dict[f'{mode}_mean_f1_list'].append(mean_f1)
        return accuracy, mean_precision, mean_recall, mean_f1, loss
    def callback_on_per_valid_epoch_end(self):
        valid_accuracy = self.count_accuracy('valid')
        if valid_accuracy > self.metrics_dict['valid_best_acc']:
            self.metrics_dict['valid_best_acc'] = valid_accuracy
            self.metrics_dict['valid_best_precision_list'] = self.metrics_dict['valid_precision_list']
            self.metrics_dict['valid_best_recall_list'] = self.metrics_dict['valid_recall_list']
            self.metrics_dict['valid_best_f1_list'] = self.metrics_dict['valid_f1_list']
            return True
        else:
            return False
    def count_accuracy(self, mode='train'):
        if not self.is_count_accuracy:
            total_num = sum(sum(self.metrics_dict[f'{mode}_confusion_matrix']))
            total_tp = sum([self.metrics_dict[f'{mode}_confusion_matrix'][i][i] for i in range(self.cls_num)])
            self.metrics_dict[f'{mode}_acc'] = round((total_tp / total_num)*100, 2)
            self.metrics_dict[f'{mode}_acc_list'].append(self.metrics_dict[f'{mode}_acc'])
            self.is_count_accuracy = True
            return self.metrics_dict[f'{mode}_acc']
        else:
            return self.metrics_dict[f'{mode}_acc']

    def count_precision(self, mode='train'):
        if not self.is_count_precision:
            for i in range(self.cls_num):
                total_num = np.sum(self.metrics_dict[f'{mode}_confusion_matrix'][:, i])
                total_tp = self.metrics_dict[f'{mode}_confusion_matrix'][i, i]
                precision = round((total_tp / total_num)*100, 2)
                self.metrics_dict[f'{mode}_precision_list'][i] = precision
            self.metrics_dict[f'{mode}_total_precision_list'].append(self.metrics_dict[f'{mode}_precision_list'])
            self.is_count_precision = True
            return self.metrics_dict[f'{mode}_precision_list']
        else:
            return self.metrics_dict[f'{mode}_precision_list']
    def count_recall(self, mode='train'):
        if not self.is_count_recall:
            for i in range(self.cls_num):
                total_num = np.sum(self.metrics_dict[f'{mode}_confusion_matrix'][i, :])
                total_tp = self.metrics_dict[f'{mode}_confusion_matrix'][i, i]
                recall = round((total_tp / total_num)*100, 2)
                self.metrics_dict[f'{mode}_recall_list'][i] = recall
            self.metrics_dict[f'{mode}_total_recall_list'].append(self.metrics_dict[f'{mode}_recall_list'])
            self.is_count_recall = True
            return self.metrics_dict[f'{mode}_recall_list']
        else:
            return self.metrics_dict[f'{mode}_recall_list']
    def count_f1(self, mode='train'):
        if not self.is_count_f1:
            precision = self.count_precision(mode)
            recall = self.count_recall(mode)
            for i in range(precision.shape[0]):
                f1 = round(2 * precision[i] * recall[i] / (precision[i] + recall[i]) / 100, 2)
                self.metrics_dict[f'{mode}_f1_list'][i] = f1
            self.metrics_dict[f'{mode}_total_f1_list'].append(self.metrics_dict[f'{mode}_f1_list'])
            self.is_count_f1 = True
            return self.metrics_dict[f'{mode}_f1_list']
        else:
            return self.metrics_dict[f'{mode}_f1_list']
    def count_loss(self, mode='train'):
        if not self.is_count_loss:
            total_num = len(self.metrics_dict[f'{mode}_preds_list'])
            self.metrics_dict[f'{mode}_loss'] = round(self.metrics_dict[f'{mode}_loss'] / total_num, 4)
            self.metrics_dict[f'{mode}_loss_list'].append(self.metrics_dict[f'{mode}_loss'])
            self.is_count_loss = True
            return self.metrics_dict[f'{mode}_loss']
        else:
            return self.metrics_dict[f'{mode}_loss']
    def count_metrics(self, mode='train'):
        return self.count_accuracy(mode), self.count_precision(mode), self.count_recall(mode), self.count_f1(mode), self.count_loss(mode)