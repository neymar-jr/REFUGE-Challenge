import numpy as np

EPS = 1e-7

def dice_coefficient(binary_segmentation, binary_gt_label):
    '''
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    '''
    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)
    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)
    # count the number of True pixels in the binary segmentation
    segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    # same for the ground truth
    gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    # same for the intersection
    intersection = float(np.sum(intersection.flatten()))
    # compute the Dice coefficient
    dice_value = 2 * intersection / (segmentation_pixels + gt_label_pixels)
    # return it
    return dice_value

def vertical_diameter(binary_segmentation):
    '''
    Get the vertical diameter from a binary segmentation.
    The vertical diameter is defined as the "fattest" area of the binary_segmentation parameter.
    Input:
        binary_segmentation: a boolean 2D numpy array representing a region of interest.
    Output:
        diameter: the vertical diameter of the structure, defined as the largest diameter between the upper and the lower interfaces
    '''
    # turn the variable to boolean, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    # get the sum of the pixels in the vertical axis
    vertical_axis_diameter = np.sum(binary_segmentation, axis=0)
    # pick the maximum value
    diameter = np.max(vertical_axis_diameter)
    # return it
    return float(diameter)

def vertical_cup_to_disc_ratio(segmentation):
    '''
    Compute the vertical cup-to-disc ratio from a given labelling map.
    The vertical cup to disc ratio is defined as here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1722393/pdf/v082p01118.pdf
    Input:
        segmentation: binary 2D numpy array representing a segmentation, with 0: optic cup, 128: optic disc, 255: elsewhere.
    Output:
        cdr: vertical cup to disc ratio
    '''
    # compute the cup diameter
    cup_diameter = vertical_diameter(segmentation==2)
    # compute the disc diameter
    disc_diameter = vertical_diameter(segmentation>0)
    return cup_diameter / (disc_diameter + EPS)

def absolute_error(predicted, reference):
    '''
    Compute the absolute error between a predicted and a reference outcomes.
    Input:
        predicted: a float value representing a predicted outcome
        reference: a float value representing the reference outcome
    Output:
        abs_err: the absolute difference between predicted and reference
    '''
    return abs(predicted - reference)
    
def evaluate_binary_segmentation(segmentation, gt_label):
    '''
    Compute the evaluation metrics of the REFUGE challenge by comparing the segmentation with the ground truth
    Input:
        segmentation: binary 2D numpy array representing the segmentation, with 2: optic cup, 1: optic disc, 0: elsewhere.
        gt_label: binary 2D numpy array representing the ground truth annotation, with the same format
    Output:
        cup_dice: Dice coefficient for the optic cup
        disc_dice: Dice coefficient for the optic disc
        cdr: absolute error between the vertical cup to disc ratio as estimated from the segmentation vs. the gt_label, in pixels
    '''
    # compute the Dice coefficient for the optic cup
    cup_dice = dice_coefficient(segmentation==2, gt_label==2)
    # compute the Dice coefficient for the optic disc
    disc_dice = dice_coefficient(segmentation>0, gt_label>0)
    # compute the absolute error between the cup to disc ratio estimated from the segmentation vs. the gt label
    cdr = absolute_error(vertical_cup_to_disc_ratio(segmentation), vertical_cup_to_disc_ratio(gt_label))
    return np.mean(cup_dice), np.mean(disc_dice), np.mean(cdr)

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
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
            self.result = evaluate_binary_segmentation(lt, lp)
    
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

        hist = self.confusion_matrix

        cup_dice, disc_dice, cdr = self.result

        disc_iou = hist[1:, 1:].sum() / (hist[1:, :].sum() + hist[:, 1:].sum() - hist[1:, 1:].sum())

        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)

        cup_iou = iou[2]

        return {
                "Disc IoU": disc_iou,
                "Cup IoU": cup_iou,
                "Mean IoU": mean_iou,
                "Mean Disc Dice": disc_dice,
                "Mean Cup Dice": cup_dice,
                "vCDR MAE": cdr,
                "Score": 0.35 * disc_dice + 0.25 * cup_dice + 0.4 * (1-cdr)
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