import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(epoch_outputs, class_names, save_path, normalize):
    """
    Save a confusion matrix plot as a PNG file using top-1 predictions.

    Args:
        epoch_outputs: list of dicts with {"preds", "labels", "valid_len"}
                       preds: (B, L, K) numpy array of top-k predictions
        class_names (list[str]): Names of classes (len = num_classes).
        save_path (str): Where to save the PNG.
        normalize (bool): If True, normalize rows to sum to 1.
    """
    y_pred = []
    y_true = []

    for o in epoch_outputs:
        preds = np.array(o["preds"])        # (B, L, K)
        labels = np.array(o["labels"])      # (B,)
        valid_len = np.array(o["valid_len"])  # (B,)

        # Top-1 at the last valid timestep
        idx = valid_len - 1
        batch_pred = preds[np.arange(len(preds)), idx, 0]  # (B,)

        y_pred.append(batch_pred)
        y_true.append(labels)

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
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
    Compute time-to-correct-decision (TTCD) normalized by sequence length.

    Args:
        epoch_outputs: list of dicts with {"preds", "labels", "valid_len"}.
                       preds: (B, L, K) per batch (numpy arrays or tensors convertible to numpy)
                       labels: (B,)
                       valid_len: (B,)
        window_size: int, stability window length.

    Returns:
        mean_nTTD: float, average normalized TTCD across all sequences.
        seq_nTTDs: list of per-sequence normalized TTCDs.
    """
    seq_nTTDs = []

    for o in epoch_outputs:
        preds = np.array(o["preds"])        # (B, L, K)
        labels = np.array(o["labels"])      # (B,)
        valid_len = np.array(o["valid_len"])  # (B,)

        for i in range(len(preds)):
            L = int(valid_len[i])
            path = preds[i, :L, 0]    # top-1 path
            label = labels[i]

            if L < window_size:
                # If too short, only "correct" if all equal to label
                if np.all(path == label):
                    seq_nTTDs.append(0.0)  # immediate decision
                else:
                    seq_nTTDs.append(1.0)  # no correct decision
                continue

            # Make sliding windows (L - window_size + 1, window_size)
            windows = np.lib.stride_tricks.sliding_window_view(path, window_shape=window_size)

            # Constant window check
            is_constant = np.all(windows == windows[:, [0]], axis=1)

            # Correct label check
            matches = is_constant & (windows[:, 0] == label)

            if np.any(matches):
                ttd = np.argmax(matches)  # earliest match
            else:
                ttd = L  # no stable correct decision

            seq_nTTDs.append(ttd / L)

    mean_nTTD = float(np.mean(seq_nTTDs)) if seq_nTTDs else 0.0
    return mean_nTTD, seq_nTTDs


def early_classification_accuracy(epoch_outputs, percents=None):
    """
    Compute early classification accuracy across percentages of observed sequence.
    """
    if percents is None:
        percents = list(range(10, 101, 10))

    acc_dict = {p: [] for p in percents}

    for o in epoch_outputs:
        preds = o["preds"]        # (B, L, K)
        labels = o["labels"]      # (B,)
        valid_len = o["valid_len"]  # (B,)

        for i in range(len(preds)):
            L = valid_len[i]
            path = preds[i, :L, 0]    # top-1 path
            label = labels[i]

            for p in percents:
                cut = max(1, int(np.floor((p / 100.0) * L)))
                pred_at_cut = path[cut - 1]
                acc_dict[p].append(int(pred_at_cut == label))

    acc_dict = {p: float(np.mean(vals)) if vals else 0.0
                for p, vals in acc_dict.items()}
    return acc_dict


def auc_over_accuracy_curve(acc_dict):
    percents = np.array(list(acc_dict.keys())) / 100.0
    accuracies = np.array(list(acc_dict.values()))

    order = np.argsort(percents)
    percents, accuracies = percents[order], accuracies[order]

    auc_value = np.trapz(accuracies, percents) / (percents[-1] - percents[0])
    return float(auc_value)



def top1_accuracy(epoch_outputs):
    """Compute Top-1 accuracy at last valid timestep (vectorized, no Python loop over samples)."""
    last_preds = []
    labels_all = []

    for o in epoch_outputs:
        preds = o["preds"]        # (B, L, K)
        labels = o["labels"]      # (B,)
        valid_len = o["valid_len"]  # (B,)

        # Extract last timestep per sequence
        idx = valid_len - 1
        last_preds.append(preds[np.arange(len(preds)), idx, 0])  # (B,)
        labels_all.append(labels)

    last_preds = np.concatenate(last_preds)   # (N,)
    labels_all = np.concatenate(labels_all)   # (N,)

    return (last_preds == labels_all).mean() if len(labels_all) > 0 else 0.0


def top5_accuracy(epoch_outputs):
    """Compute Top-5 accuracy at last valid timestep (vectorized)."""
    last_topk = []
    labels_all = []

    for o in epoch_outputs:
        preds = o["preds"]        # (B, L, K)
        labels = o["labels"]      # (B,)
        valid_len = o["valid_len"]  # (B,)

        idx = valid_len - 1
        last_topk.append(preds[np.arange(len(preds)), idx])  # (B, K)
        labels_all.append(labels)

    last_topk = np.concatenate(last_topk, axis=0)   # (N, K)
    labels_all = np.concatenate(labels_all, axis=0) # (N,)

    return (last_topk == labels_all[:, None]).any(axis=1).mean() if len(labels_all) > 0 else 0.0