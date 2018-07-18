import random

class RSub:
  def __init__(self):
    pass

  def get_subset(self, X, X_lab, K):
    indices = sorted( random.sample(range(len(X)), K) )
    X_sub = [X[i] for i in indices]
    X_lab_sub = [X_lab[i] for i in indices]
    return X_sub, X_lab_sub

