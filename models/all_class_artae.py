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
  def __init__(self, n_channel, w_img):
    self.name = "AEnet"
    self.n_channel, self.w_img = n_channel, w_img

  def save(self, loc):
    torch.save(self.ae.state_dict(), loc+".mdl")

  def load(self, loc):
    ae = AE().cuda()
    ae.load_state_dict(torch.load(loc+".mdl"))
    self.ae = ae

  def torchify(self, X):
    ret =  to_torch(X, "float").view(-1, self.n_channel, self.w_img, self.w_img)
    return ret

  def learn_ae(self, X):
    ae = AE().cuda()
    # Just go over all the data batch for now
    for i in range(len(X)):
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

  def tsne(self, embedded_X, labels=None):
    cl_colors = np.linspace(0, 1, len(labels)) if (labels is not None) else ['blue']
    from sklearn.manifold import TSNE
    X_tsne = TSNE(n_components=2).fit_transform(embedded_X)
    import matplotlib
    # matplotlib.use("svg")
    x = [x[0] for x in X_tsne]
    y = [x[1] for x in X_tsne]
    colors = [cl_colors[lab] for lab in labels] if (labels is not None) else 'blue'
    plt.scatter(x, y, c=colors, alpha=0.5)
    plt.savefig('drawings/2d_tsne_ae_'+'.png')
    plt.clf()
    return X_tsne

  def make_knn(self, X, Y, embed=True):
    X_emb = self.embed(X) if embed else X
    from sklearn import neighbors
    # I HRD SOMEWHERE THAT USING k = LOG(len(DATA)) is good
    clf = neighbors.KNeighborsClassifier(1+int(np.log(len(Y))), weights='distance')
    knn_cl = clf.fit(X_emb, Y)
    def classify(X_new):
      X_new_emb = self.embed(X_new) if embed else X_new
      return knn_cl.predict(X_new_emb)
    return classify

  def sub_select1(self, n_samples, X, Y, embed=True):
    steps = min(100, n_samples-2)
    n_batch = max(n_samples // steps, 1)
    # TODO: initialize with clusters
    r_idxs = np.random.choice(np.arange(len(X)), n_batch, replace=False)
    X_sub = X[r_idxs]
    Y_sub = Y[r_idxs]
    while len(Y_sub) < n_samples:
      clf = self.make_knn(X_sub, Y_sub, embed)
      pred = clf(X)
      incorrect = (pred != Y)
      X_remain = X[incorrect]
      Y_remain = Y[incorrect]

      print ("misclasified", len(X_remain) )

      r_idxs = np.random.choice(np.arange(len(X_remain)), n_batch, replace=False)
      X_add = X_remain[r_idxs]
      Y_add = Y_remain[r_idxs]
      X_sub = np.concatenate((X_sub, X_add))
      Y_sub = np.concatenate((Y_sub, Y_add))

    return X_sub, Y_sub

  def sub_select2(self, n_samples, X, Y, embed=True, inc_size=10, search_width=10):
    def loss(X_sub, Y_sub):
      clf = self.make_knn(X_sub, Y_sub, embed)
      pred = clf(X)
      incorrect = (pred != Y)
      X_remain = X[incorrect]
      Y_remain = Y[incorrect]
      return sum(incorrect), X_remain, Y_remain

    # get a batch of new samples from remaining set
    def make_sample(X_rem, Y_rem, inc_size):
      r_idxs = np.random.choice(np.arange(len(X_rem)), inc_size, replace=False)
      return X_rem[r_idxs], Y_rem[r_idxs]

    def one_step(X_sub, Y_sub, inc_size, search_width):
      samples = [make_sample(X, Y, inc_size) for _ in range(search_width)]
      cand_sub = [(np.concatenate((X_sub, samp[0])), np.concatenate((Y_sub, samp[1]))) for samp in samples] if len(X_sub) > 0 else samples

      loss_cand = [(loss(*cand)[0], cand) for cand in cand_sub]
      best_score, best_cand = min(loss_cand, key = lambda t: t[0])
      print (best_score)
      return best_cand

    X_sub, Y_sub = [], []
    for iter_n in range(n_samples // inc_size):
      print (iter_n)
      X_sub, Y_sub = one_step(X_sub, Y_sub, inc_size, search_width) 

    return X_sub, Y_sub
    
  def sub_select3(self, n_samples, X, Y, embed=True, inc_size=10, search_width=10):
    dec_size = inc_size // 2
    def loss(X_sub, Y_sub):
      clf = self.make_knn(X_sub, Y_sub, embed)
      pred = clf(X)
      incorrect = (pred != Y)
      X_remain = X[incorrect]
      Y_remain = Y[incorrect]
      return sum(incorrect), X_remain, Y_remain

    # get a batch of new samples from remaining set
    def make_sample(X_rem, Y_rem, inc_size):
      r_idxs = np.random.choice(np.arange(len(X_rem)), inc_size, replace=False)
      return X_rem[r_idxs], Y_rem[r_idxs]

    def one_step_add(X_sub, Y_sub):
      samples = [make_sample(X, Y, inc_size) for _ in range(search_width)]
      cand_sub = [(np.concatenate((X_sub, samp[0])), np.concatenate((Y_sub, samp[1]))) for samp in samples] if len(X_sub) > 0 else samples

      loss_cand = [(loss(*cand)[0], cand) for cand in cand_sub]
      best_score, best_cand = min(loss_cand, key = lambda t: t[0])
      return best_score, best_cand

    def one_step_sub(X_sub, Y_sub):
      red_size = len(X_sub) - dec_size
      cand_redux = [make_sample(X_sub, Y_sub, red_size) for _ in range(search_width)]

      loss_cand = [(loss(*cand)[0], cand) for cand in cand_redux]
      best_score, best_cand = min(loss_cand, key = lambda t: t[0])
      return best_score, best_cand

    X_sub, Y_sub = [], []
    while len(X_sub) < n_samples:
      add_score, (X_sub, Y_sub) = one_step_add(X_sub, Y_sub)
      sub_score, (X_sub, Y_sub) = one_step_sub(X_sub, Y_sub) 
      print (len(Y_sub), add_score, sub_score)

    return X_sub, Y_sub


def AEnet_Maker(n_channel, w_img):
  def call():
    return AEnet(n_channel, w_img)
  return call

if __name__ == '__main__':
  print ("hi")
  import pickle
  LOC = "./data/artificial1/artificial1.p"
  X,Y = pickle.load(open(LOC,"rb"))
  X_tr, Y_tr = X[:60000], Y[:60000]

  aenet = AEnet_Maker(1, 28)()
  aenet.learn_ae(X_tr)
  aenet.save('saved_models/artificial_allae_model')



