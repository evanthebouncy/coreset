import random
from sklearn.linear_model import LogisticRegression
import numpy as np

class LRegrActive:

  def __init__(self):
    self.name = "LRegrActive"

  # learn and return a logstic regression model from a corpus
  def learn(self, train_corpus):
    train_data, train_label = train_corpus
    logisticRegr = LogisticRegression()
    logisticRegr.fit(train_data, train_label)
    return logisticRegr

  # takes in a logistic regression model and a corpus
  # return the top-K most high-entropy items
  def solicit(self, lregr, corpus, K):
    X, y = corpus
    # compute probability and entropy of the input X
    probs = lregr.predict_proba(X)
    entrops = np.sum( -probs * np.log(probs), axis=1)
    # select the top K entries index and return the entries there
    ind = np.argpartition(entrops, -K)[-K:]
    return X[ind], y[ind]

  def get_subset(self, X, X_lab, K):
    # first select 1 / 10 at random
    indices = sorted( random.sample(range(len(X)), K // 10) )
    X_sub =     np.array([X[i] for i in indices])
    X_lab_sub = np.array([X_lab[i] for i in indices])

    for i in range(9):
      lregr = self.learn((X_sub, X_lab_sub))
      more_X, more_y = self.solicit(lregr, (X, X_lab), K // 10)
      X_sub = np.concatenate((X_sub, more_X))
      X_lab_sub = np.concatenate((X_lab_sub, more_y))

    return X_sub, X_lab_sub

