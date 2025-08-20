import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(epoch_outputs, class_names, save_path, normalize):
    """
    Save a confusion matrix plot as a PNG file using top-1 predictions.

    Args:
        epoch_outputs: list of dicts with {"preds", "labels", "valid_len"}
                       preds: (B, T, K) numpy array of top-k predictions
        class_names (list[str]): Names of classes (len = num_classes).
        save_path (str): Where to save the PNG.
        normalize (bool): If True, normalize rows to sum to 1.
    """
    preds_all = np.concatenate([o["preds"] for o in epoch_outputs], axis=0)  # (N, L, K)
    labels_all = np.concatenate([o["labels"] for o in epoch_outputs], axis=0)
    valid_len_all = np.concatenate([o["valid_len"] for o in epoch_outputs], axis=0)

    # Top-1 prediction at the last valid timestep
    y_pred = []
    for i in range(preds_all.shape[0]):
        y_pred.append(preds_all[i, valid_len_all[i] - 1, 0])
    y_pred = np.array(y_pred)

    y_true = labels_all

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # avoid NaNs if row sum = 0

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    # Title & axis labels
    ax.set_title("Confusion Matrix (Top-1 Predictions)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Annotate each cell
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def time_to_correct_decision(epoch_outputs, window_size: int = 3):
    """
    Compute time-to-correct-decision normalized by sequence length.

    Args:
        epoch_outputs: list of dicts with {"preds", "labels", "valid_len"}
                       preds: (N, L, K)
        window_size: int, stability window length.
    Returns:
        mean_nTTD: float, average normalized time-to-correct-decision across sequences.
        seq_nTTDs: list of per-sequence normalized times.
    """
    preds_all = np.concatenate([o["preds"] for o in epoch_outputs], axis=0)   # (N, L, K)
    labels_all = np.concatenate([o["labels"] for o in epoch_outputs], axis=0)
    valid_len_all = np.concatenate([o["valid_len"] for o in epoch_outputs], axis=0)

    N = preds_all.shape[0]
    seq_nTTDs = []

    for i in range(N):
        preds = preds_all[i, :valid_len_all[i], 0]  # take top-1 path
        label = labels_all[i]
        L = valid_len_all[i]

        ttd = L  # default: no correct stable decision
        for t in range(L - window_size + 1):
            window = preds[t:t+window_size]
            if np.all(window == window[0]) and window[0] == label:
                ttd = t
                break
        seq_nTTDs.append(ttd / L)

    return float(np.mean(seq_nTTDs)), seq_nTTDs


def early_classification_accuracy(epoch_outputs, percents=None):
    """
    Compute early classification accuracy across percentages of observed sequence.

    Args:
        epoch_outputs: list of dicts with {"preds", "labels", "valid_len"}.
                       preds: (N, L, K)
        percents: list of integers (1–100). Default = 10,20,...,100
    Returns:
        acc_dict: dict {percent: accuracy}
    """
    if percents is None:
        percents = list(range(10, 101, 10))  # 10,20,...,100

    preds_all = np.concatenate([o["preds"] for o in epoch_outputs], axis=0)   # (N, L, K)
    labels_all = np.concatenate([o["labels"] for o in epoch_outputs], axis=0)
    valid_len_all = np.concatenate([o["valid_len"] for o in epoch_outputs], axis=0)

    acc_dict = {p: [] for p in percents}

    N = preds_all.shape[0]
    for i in range(N):
        preds = preds_all[i, :valid_len_all[i], 0]  # top-1 path
        label = labels_all[i]
        L = valid_len_all[i]

        for p in percents:
            cut = max(1, int(np.floor((p / 100.0) * L)))
            pred_at_cut = preds[cut - 1]
            acc_dict[p].append(int(pred_at_cut == label))

    # average per percent
    acc_dict = {p: float(np.mean(vals)) for p, vals in acc_dict.items()}
    return acc_dict


def auc_over_accuracy_curve(acc_dict):
    """
    Compute area under accuracy-observation curve (trapezoid rule).

    Args:
        acc_dict: dict {percent: accuracy}, from early_classification_accuracy.
    Returns:
        auc_value: float
    """
    percents = np.array(list(acc_dict.keys())) / 100.0  # convert back to fractions
    accuracies = np.array(list(acc_dict.values()))

    # sort by percent
    order = np.argsort(percents)
    percents, accuracies = percents[order], accuracies[order]

    auc_value = np.trapz(accuracies, percents) / (percents[-1] - percents[0])
    return float(auc_value)


def top1_accuracy(epoch_outputs):
    """Compute Top-1 accuracy at last valid timestep."""
    preds_all = np.concatenate([o["preds"] for o in epoch_outputs], axis=0)   # (N, L, K)
    labels_all = np.concatenate([o["labels"] for o in epoch_outputs], axis=0)
    valid_len_all = np.concatenate([o["valid_len"] for o in epoch_outputs], axis=0)

    N = preds_all.shape[0]
    correct = 0
    for i in range(N):
        last_pred = preds_all[i, valid_len_all[i] - 1, 0]  # top-1
        if last_pred == labels_all[i]:
            correct += 1

    return correct / N if N > 0 else 0.0


def top5_accuracy(epoch_outputs):
    """Compute Top-5 accuracy at last valid timestep."""
    preds_all = np.concatenate([o["preds"] for o in epoch_outputs], axis=0)   # (N, L, K)
    labels_all = np.concatenate([o["labels"] for o in epoch_outputs], axis=0)
    valid_len_all = np.concatenate([o["valid_len"] for o in epoch_outputs], axis=0)

    N = preds_all.shape[0]
    correct = 0
    for i in range(N):
        last_topk = preds_all[i, valid_len_all[i] - 1]  # shape (K,)
        if labels_all[i] in last_topk:
            correct += 1

    return correct / N if N > 0 else 0.0
