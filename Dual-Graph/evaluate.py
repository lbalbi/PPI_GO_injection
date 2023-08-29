from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
import sklearn.metrics
import pandas as pd
import os
import numpy as np
import warnings
try:
    import torch
except ImportError:
    torch = None

warnings.filterwarnings('ignore')

### Evaluator for node property prediction
class Evaluator:
    def __init__(self, name, nlabels):
        self.name = name

        meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col = 0)
        if not self.name in meta_info:
            print(self.name)
            error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
            error_mssg += 'Available datasets are as follows:\n'
            error_mssg += '\n'.join(meta_info.keys())
            raise ValueError(error_mssg)

        #self.num_tasks = int(meta_info[self.name]['num tasks'])
        self.num_tasks = nlabels
        self.eval_metric = meta_info[self.name]['eval metric']
        #print(self.eval_metric)


    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'rocauc' or self.eval_metric == 'acc':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_node, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_node, num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.num_tasks, y_true.shape[1]))

            return y_true, y_pred

        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))


    def eval(self, input_dict):

        if self.eval_metric == 'rocauc':
            y_true, y_pred = self._parse_and_check_input(input_dict) # ambas c/ matriz de nº de nodes por nº de labels          
            #recall = self._eval_recall(y_true, y_pred, "numpy")
            #precision = self._eval_precision(y_true, y_pred, "numpy")
            #accuracy = self._eval_acc(y_true, y_pred)
            return {'rocauc': self._eval_rocauc(y_true, y_pred), 'acc': self._eval_acc(y_true, y_pred)}   #, 'recall': recall['recall'], 'precision': precision['precision']}
            #return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == 'acc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc(y_true, y_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_node, num_task)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_node, num_task)\n'
            desc += 'where y_pred stores score values (for computing ROC-AUC),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        elif self.eval_metric == 'acc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_node, num_task)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_node, num_task)\n'
            desc += 'where y_pred stores predicted class label (integer),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'rocauc\': rocauc}\n'
            desc += '- rocauc (float): ROC-AUC score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'acc':
            desc += '{\'acc\': acc}\n'
            desc += '- acc (float): Accuracy score averaged across {} task(s)\n'.format(self.num_tasks)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC and AP score averaged across tasks
        '''
        rocauc_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))
                #rocauc_list = roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i])

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')
        #print(len(rocauc_list))
        #return rocauc_list
        return sum(rocauc_list)/len(rocauc_list)
        #return {'rocauc': sum(rocauc_list)/len(rocauc_list)}

    def _eval_acc(self, y_true, y_pred):
        acc_list = []
        y_pred1 = sklearn.preprocessing.normalize(y_pred)
        for i in range(y_true.shape[1]):
            #y_true_list = y_true[:,i].tolist()
            #y_pred_list = [ 1 if m > 0 else 0 for m in y_pred[:,i]]
            y_pred_list = (y_pred1[:,i] > 0).astype(np.int_)
            acc_list.append(sklearn.metrics.accuracy_score(y_true[:,i],y_pred_list))
            #acc_list = sklearn.metrics.accuracy_score(y_true[:,i],y_pred_list)
        return sum(acc_list)/len(acc_list)


    def _eval_recall(self,y_true_pos,y_pred_pos, type_info):
        if type_info == 'torch':
            ones = np.ones(len(y_pred_pos))
            zeros = np.zeros(len(y_pred_neg))
            y_true = np.append(ones,zeros)
            dict1={}
            dict1["recall"]=y_pred_pos.numpy()
            dict1["recall"]= np.append(dict1["recall"], y_pred_neg.numpy())
            dict1["recall"] = [ 1 if m > 0 else 0 for m in dict1["recall"]]
            recall = sklearn.metrics.recall_score(y_true,dict1["recall"])
        else:
        # #if type_info is numpy:
        #     ones = np.ones(len(y_pred_pos))
        #     zeros = np.zeros(len(y_pred_pos))
        #     y_true = np.append(ones,zeros)
        #     y_pred= np.append(y_pred_pos, y_pred_pos)
        #     print(y_pred)
        #     print(len(y_pred))
        #     #print(len(y_pred[0]))
        #     y_pred = [ 1 if m > 0.5 else 0 for m in y_pred]
            recall_lst = []
            #print(len(y_true_pos))
            #print(len(y_pred_pos))
            for i in range(y_true_pos.shape[1]):
                y_pred = [ 1 if m > 0 else 0 for m in y_pred_pos[:][i]]
                #print(len(y_true_pos[:][i]))
                #print(len(y_pred))
                recall = sklearn.metrics.recall_score(y_true_pos[:][i],y_pred) #, zero_division=0)
                recall_lst.append(recall)
        #print(len(recall_lst))
        return {'recall': recall_lst}


    def _eval_precision(self,y_true_pos, y_pred_pos, type_info):
        if type_info == 'torch':
            ones = np.ones(len(y_pred_pos))
            zeros = np.zeros(len(y_pred_neg))
            y_true = np.append(ones,zeros)
            dict2={}
            dict2["precision"]=y_pred_pos.numpy()
            dict2["precision"]= np.append(dict2["precision"], y_pred_neg.numpy())
            dict2["precision"] = [ 1 if m > 0 else 0 for m in dict2["precision"]]
            precision = sklearn.metrics.precision_score(y_true,dict2["precision"])
        else:
        # if type_info is numpy
            # ones = np.ones(len(y_pred_pos))
            # zeros = np.zeros(len(y_pred_neg))
            # y_true = np.append(ones,zeros)
            # y_pred = np.append(y_pred_pos, y_pred_neg)
            precision_lst = []
            for i in range(y_true_pos.shape[1]):
                y_pred = [ 1 if m > 0 else 0 for m in y_pred_pos[:][i]]
                #print(len(y_true_pos[:][i]))
                #print(len(y_pred))
                precision = sklearn.metrics.precision_score(y_true_pos[:][i],y_pred) #, zero_division=0)
                precision_lst.append(precision)
        #print(len(precision_lst))
        return {'precision': precision_lst}




if __name__ == '__main__':
    ### rocauc case



    evaluator = Evaluator('ogbn-proteins')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = torch.tensor(np.random.randint(2, size = (100,112)))
    y_pred = torch.tensor(np.random.randn(100,112))
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)

    ### acc case
    evaluator = Evaluator('ogbn-products')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = np.random.randint(5, size = (100,1))
    y_pred = np.random.randint(5, size = (100,1))
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)

    ### acc case
    evaluator = Evaluator('ogbn-arxiv')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = np.random.randint(5, size = (100,1))
    y_pred = np.random.randint(5, size = (100,1))
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)



