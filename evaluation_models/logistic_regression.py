from sklearn.linear_model import LogisticRegression
import numpy as np

class LRegr:

  def __init__(self):
    self.name = "LRegr"

  def learn(self, train_corpus):
    train_data, train_label = train_corpus
    logisticRegr = LogisticRegression()
    logisticRegr.fit(train_data, train_label)
    self.logisticRegr = logisticRegr

  def evaluate(self, test_corpus):
    test_data, test_label = test_corpus
    label_pred = self.logisticRegr.predict(test_data)
    return np.sum(label_pred == test_label) / len(test_label)

if __name__ == "__main__":
  X = np.array( [[0, 0], [1, 1]] )
  Y = np.array( [0, 1] )

  lregr = LRegr()
  lregr.learn((X,Y))
  print ( lregr.evaluate((X,Y)) )

