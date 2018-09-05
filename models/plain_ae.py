import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

# for drawing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def to_torch(x, dtype, req = False):
  tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x

class AE(nn.Module):
  def __init__(self):
    super(AE, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
      nn.ReLU(True),
      nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
      nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
      nn.ReLU(True),
      nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
      nn.Sigmoid(),
    )
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
      nn.ReLU(True),
      nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
      nn.Tanh()
    )
    self.opt = torch.optim.Adam(self.parameters(), lr=0.0002)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

  def save(self, loc):
    torch.save(net.state_dict(), loc) 

class AEnet():
  # takes in channel variable n and width of image
  def __init__(self, n_channel, w_img, class_id):
    self.name = "AEnet"
    self.n_channel, self.w_img = n_channel, w_img
    self.class_id = class_id

  def save(self, loc):
    torch.save(self.ae.state_dict(), loc+"_"+str(self.class_id)+".mdl")

  def load(self, loc, class_id):
    ae = AE().cuda()
    ae.load_state_dict(torch.load(loc+"_"+str(class_id)+".mdl"))
    self.ae = ae

  def torchify(self, X):
    return to_torch(X, "float").view(-1, self.n_channel, self.w_img, self.w_img)

  def learn_ae(self, X):
    ae = AE().cuda()
    for i in range(len(X) * 20):
      # load in the datas
      indices = sorted( random.sample(range(len(X)), 40) )
      X_sub = np.array([X[i] for i in indices])
      # convert to proper torch forms
      X_sub = self.torchify(X_sub)

      # optimize 
      ae.opt.zero_grad()
      output = ae(X_sub)
      loss_fun = nn.MSELoss()
      loss = loss_fun(output, X_sub)
      loss.backward()
      ae.opt.step()

      if i % 4000 == 0:
        print (X_sub.size())
        print (output.size())
        print (loss)

        X_sub_np = X_sub[0].data.cpu().view(28,28).numpy()
        output_np = output[0].data.cpu().view(28,28).numpy()
        plt.imsave('test1.png', X_sub_np)
        plt.imsave('test2.png', output_np)

    self.ae = ae
    return ae

  def embed(self, X):
    X = np.array(X)
    X = self.torchify(X)
    encoded = self.ae.encoder(X).view(-1, 8*2*2)
    return encoded.data.cpu().numpy()

  def tsne(self, embedded_X):
    from sklearn.manifold import TSNE
    X_tsne = TSNE(n_components=2).fit_transform(embedded_X)
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    x = [x[0] for x in X_tsne]
    y = [x[1] for x in X_tsne]
    plt.scatter(x, y, alpha=0.5)
    plt.savefig('drawings/2d_tsne_'+str(self.class_id)+'.png')
    plt.clf()
    print (X_tsne)


def AEnet_Maker(n_channel, w_img):
  def call(class_id):
    return AEnet(n_channel, w_img, class_id)
  return call




