from sklearn.metrics import f1_score

def simple_accuracy(preds, labels):
    assert len(preds) == len(labels)
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc =simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def snli_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "snli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)