import random

class RSub:
  def __init__(self):
    self.name = "Rand Subset"

  def get_subset(self, X, y, K):
    indices = sorted( random.sample(range(len(X)), K) )
    X_sub = [X[i] for i in indices]
    y_sub = [y[i] for i in indices]
    return X_sub, y_sub

  def get_subset_balanced(self, X, y, K, n_labels):
    assert K % n_labels == 0, "cannot be balanced blyat"

    X_sub, y_sub = [], []
    for lab in range(n_labels):
      # sub-sample a small set from 60k
      sample_indexs = sorted( random.sample(range(len(X)), K * 10) )
      y_sample = y[sample_indexs]
      lab_idxs = [idx for idx in range(len(y_sample)) if y_sample[idx] == lab]
      lab_idxs  = random.sample(lab_idxs, K // n_labels)
      X_sub_lab = [X[i] for i in lab_idxs]
      y_sub_lab = [y[i] for i in lab_idxs]
      X_sub.extend(X_sub_lab)
      y_sub.extend(y_sub_lab)

    return X_sub, y_sub




