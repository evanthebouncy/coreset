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
  def __init__(self):
    super(CNN, self).__init__()
    # 1 input channel, 10 output channel, 5x5 sliding window
    self.conv1 = nn.Conv2d(1, 10, 5)
    self.conv2 = nn.Conv2d(10, 20, 5)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)
    self.opt = torch.optim.Adam(self.parameters(), lr=0.0002)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

class Cnet():
  def __init__(self):
    self.name = "Cnet"

  def learn(self, train_corpus):
    cnn = CNN().cuda()
    # cnn = CNN().cuda()
    X, y = train_corpus
    for i in range(len(y) // 40 * 10):
      # load in the datas
      indices = sorted( random.sample(range(len(X)), 40) )
      X_sub = np.array([X[i] for i in indices])
      y_sub = np.array([y[i] for i in indices])
      # convert to proper torch forms
      X_sub = to_torch(X_sub, "float").view(-1, 1, 28, 28)
      y_sub = to_torch(y_sub, "int")

      # optimize 
      cnn.opt.zero_grad()
      output = cnn(X_sub)
      loss = F.nll_loss(output, y_sub)
      loss.backward()
      cnn.opt.step()

    self.cnn = cnn

  def evaluate(self, test_corpus):
    test_X, test_y = test_corpus
    # convert to proper torch forms
    test_X = to_torch(test_X, "float").view(-1, 1, 28, 28)
    output = self.cnn(test_X)
    y_pred = np.argmax( output.data.cpu().numpy(), axis=1 )
    return np.sum(y_pred == test_y) / len(test_y)




