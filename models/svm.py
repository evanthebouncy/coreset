from sklearn import svm
import numpy as np

class Svmm:

  def __init__(self):
    self.name = "Svmm"

  def learn(self, train_corpus):
    train_data, train_label = train_corpus
    clf = svm.SVC()
    clf.fit(train_data, train_label)
    self.clf = clf

  def evaluate(self, test_corpus):
    test_data, test_label = test_corpus
    label_pred = self.clf.predict(test_data)
    return np.sum(label_pred == test_label) / len(test_label)

if __name__ == "__main__":
  X = np.array( [[0, 0], [1, 1]] )
  Y = np.array( [0, 1] )

  svmm = Svmm()
  svmm.learn((X,Y))
  print ( svmm.evaluate((X,Y)) )

