from sklearn import tree
import numpy as np


class DTree:

  def __init__(self):
    self.name = "DTree"

  def learn(self, train_corpus):
    train_data, train_label = train_corpus
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_label)
    self.clf = clf

  def evaluate(self, test_corpus):
    test_data, test_label = test_corpus
    label_pred = self.clf.predict(test_data)
    return np.sum(label_pred == test_label) / len(test_label)


if __name__ == "__main__":
  X = np.array( [[0, 0], [1, 1]] )
  Y = np.array( [0, 1] )

  dtree = DTree()
  dtree.learn((X,Y))
  print ( dtree.evaluate((X,Y)) )
