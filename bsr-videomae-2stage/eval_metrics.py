from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, balanced_accuracy_score, f1_score, average_precision_score
import numpy as np
# Reference:
# https://pyhealth.readthedocs.io/en/latest/_modules/pyhealth/metrics/binary.html
# https://pyhealth.readthedocs.io/en/latest/_modules/pyhealth/metrics/multiclass.html

# Not using pyhealth for a wired error:
# FileExistsError: [Errno 17] File exists: '/root/.cache/pyhealth/'

def get_metrics_binary(true, pred, threshold=0.5):
    prob = pred.copy()
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    score = {}
    score["pr_auc"] = average_precision_score(true, prob)
    score["roc_auc"] = roc_auc_score(true, prob)
    score["balanced_accuracy"] = balanced_accuracy_score(true, pred)
    return score
    
def get_metrics_multiclass(true, pred, all_classes=[0, 1]):
    pred = np.argmax(pred, axis=-1)
    score = {}
    score["balanced_accuracy"] = balanced_accuracy_score(true, pred)
    score["cohen_kappa"] = cohen_kappa_score(true, pred, labels=all_classes)
    score["f1_weighted"] = f1_score(true, pred, average="weighted", labels=all_classes)
    return score
