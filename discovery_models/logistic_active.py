import random
from sklearn.linear_model import LogisticRegression
import numpy as np

class LRegrActive:

  def __init__(self, criteria):
    self.name = "LRegrActive"
    self.criteria = criteria

  # learn and return a logstic regression model from a corpus
  def learn(self, train_corpus):
    train_data, train_label = train_corpus
    logisticRegr = LogisticRegression(solver='sag')
    logisticRegr.fit(train_data, train_label)
    return logisticRegr

  # return either negative entropy or class_prob depend on criteria
  def solicit_criteria(self, probs, y):
    assert self.criteria in ["max_ent", "min_prob"]
    if self.criteria == "max_ent":
      entrops = np.sum( -probs * np.log(probs), axis=1)
      return -entrops
    if self.criteria == "min_prob":
      class_prob = probs[range(len(y)), y]
      return class_prob

  # takes in a logistic regression model and a corpus
  # return the top-K most high-entropy items
  def solicit(self, lregr, corpus, K):
    X, y = corpus
    # compute probability and entropy of the input X
    probs = lregr.predict_proba(X)
    scores = self.solicit_criteria(probs, y)
    # select the top K entries index and return the entries there
    ind = np.argpartition(scores, K)[:K]
    return X[ind], y[ind]

  # make sure the initial subset cover all the classes
  def initial_subset(self, X, y, kk):
    indices = sorted( random.sample(range(len(X)), kk) )
    X_sub = np.array([X[i] for i in indices])
    y_sub = np.array([y[i] for i in indices])
    # check we have 1 data for every class, if not try again with bigger subset
    for yy in y:
      if yy not in y_sub:
        return self.initial_subset(X, y, kk+1)
    return X_sub, y_sub

  def get_subset(self, X, X_lab, K):
    # first select 1 / 10 at random
    X_sub, X_lab_sub = self.initial_subset(X, X_lab, K // 10)

    while len(X_lab_sub) < K:
      lregr = self.learn((X_sub, X_lab_sub))
      more_X, more_y = self.solicit(lregr, (X, X_lab), K // 10)
      X_sub = np.concatenate((X_sub, more_X))
      X_lab_sub = np.concatenate((X_lab_sub, more_y))
  
    print ("final size ", len(X_lab_sub))
    return X_sub, X_lab_sub

