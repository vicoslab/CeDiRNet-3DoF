
import numpy as np
import json

from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_AP_and_F1(_Y, _P, ignore_mask=None):
    _Y = np.squeeze(np.array(_Y))
    _P = np.squeeze(np.array(_P))

    # replace inf values (missing detection) with something that is smaller than any other valid probability in P
    missing_det_mask = _P == -np.inf
    if any(missing_det_mask):
        _P[missing_det_mask] = np.min(_P[~missing_det_mask]) - 1

    if ignore_mask is not None:
        ignore_mask = np.squeeze(np.array(ignore_mask))

        _Y = _Y[ignore_mask == 0]
        _P = _P[ignore_mask == 0]

    roc_curve = precision_recall_curve(_Y, _P)

    precision = roc_curve[0]
    recall = roc_curve[1]
    thrs = roc_curve[2]

    # remove missed detections from precision-recall scores (if there are any)
    # this is needed to prevent counting recall=100% when recall was never 100%
    if any(missing_det_mask):
        precision = precision[1:]
        recall = recall[1:]
        thrs = thrs[1:]

    # do not call average_precision_score(Y,P) directly since it will not be able to count
    # missed detection as proprelly missed
    AP = -np.sum(np.diff(recall) * np.array(precision)[:-1])

    F1 = 2 * (precision * recall) / (precision + recall)

    return AP, F1, precision, recall, thrs
