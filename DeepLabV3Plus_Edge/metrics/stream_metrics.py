import numpy as np
from sklearn import metrics


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            # self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
            # self.f1_score = metrics.f1_score(lt.flatten(), lp.flatten())
            self.auc = metrics.roc_auc_score(lt.flatten(), lp.flatten())
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            string += "%s: %f\n"%(k, v)

        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):

        # hist = self.confusion_matrix

        auc = self.auc

        # f1_score = self.f1_score

        # tp = hist[1][1]
        # fp = hist[0][1]
        # fn = hist[1][0]

        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)

        # f_1 = 2 * precision * recall / (precision + recall)

        # beta = 0.5
        # f_beta = (1 + pow(beta, 2)) * precision * recall / (pow(beta, 2) * precision + recall)

        return {
                # "sklearn_f_1": f1_score,
                # "f_1": f_1,
                # "f_beta": f_beta,
                "auc": auc,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
