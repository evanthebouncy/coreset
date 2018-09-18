from sklearn import svm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

def to_torch(x, dtype, req = False):
  tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x

class CNN(nn.Module):
  def __init__(self, n_channel, w_img):
    super(CNN, self).__init__()

    n_hiddens = {
          (1, 28) : 320,
          (3, 32) : 500,
        }
    self.n_hidden = n_hiddens[(n_channel, w_img)] 
    # 1 input channel, 10 output channel, 5x5 sliding window
    self.conv1 = nn.Conv2d(n_channel, 10, 5)
    self.conv2 = nn.Conv2d(10, 20, 5)

    self.fc1 = nn.Linear(self.n_hidden, 50)
    self.fc2 = nn.Linear(50, 10)
    self.opt = torch.optim.Adam(self.parameters(), lr=0.0002)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = x.view(-1, self.n_hidden)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

class Cnet():
  # takes in channel variable n and width of image
  def __init__(self, n_channel, w_img):
    self.name = "Cnet"
    self.n_channel, self.w_img = n_channel, w_img

  def torchify(self, X):
    return to_torch(X, "float").view(-1, self.n_channel, self.w_img, self.w_img)

  def learn(self, train_corpus):
    cnn = CNN(self.n_channel, self.w_img).cuda()
    # cnn = CNN().cuda()
    X, y = train_corpus

    losses = []
    while True:
      # load in the datas
      b_size = min(40, len(X) // 2)
      indices = sorted( random.sample(range(len(X)), b_size) )
      X_sub = np.array([X[i] for i in indices])
      y_sub = np.array([y[i] for i in indices])
      # convert to proper torch forms
      X_sub = self.torchify(X_sub)
      y_sub = to_torch(y_sub, "int")

      # optimize 
      cnn.opt.zero_grad()
      output = cnn(X_sub)
      loss = F.nll_loss(output, y_sub)
      losses.append( loss.data.cpu().numpy() )
      # terminate if no improvement
      if loss < 1e-4 or min(losses) < min(losses[-1000:]):
        break
      loss.backward()
      cnn.opt.step()

    self.cnn = cnn
    return cnn

  def evaluate(self, test_corpus):
    test_X, test_y = test_corpus
    # convert to proper torch forms
    test_X = self.torchify(test_X)
    output = self.cnn(test_X)
    y_pred = np.argmax( output.data.cpu().numpy(), axis=1 )
    return np.sum(y_pred == test_y) / len(test_y)


  # ===================== SUBSET SELECTION ======================

  # takes in a logistic regression model and a corpus
  # return the top-K most high-entropy items
  def solicit(self, trained_model, corpus, K):
    X, y = corpus
    X_torch = self.torchify(X)
    # compute probability and entropy of the input X
    output = trained_model(X_torch)
    probs  = output.data.cpu().numpy()
    scores = probs[range(len(y)), y]
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
      trained_model = self.learn((X_sub, X_lab_sub))
      more_X, more_y = self.solicit(trained_model, (X, X_lab), K // 10)
      X_sub = np.concatenate((X_sub, more_X))
      X_lab_sub = np.concatenate((X_lab_sub, more_y))
  
    print ("final size ", len(X_lab_sub))
    return X_sub, X_lab_sub

  def get_subset_rank(self, X, y):
    trained_model = self.learn((X, y))

    X_torch = self.torchify(X)
    # compute probability and entropy of the input X
    output = trained_model(X_torch)
    probs  = output.data.cpu().numpy()
    y_pred = np.argmax(probs, axis=1)
    correct = y == y_pred
    scores = probs[range(len(y)), y]
    score_data = list(zip(scores, X, y, correct))
    score_data = filter(lambda x:x[3], score_data)
    score_data = [x[:3] for x in score_data]
    return sorted(score_data, key=lambda x:x[0])


def Cnet_Maker(n_channel, w_img):
  def call():
    return Cnet(n_channel, w_img)
  return call



