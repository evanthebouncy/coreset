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
dtype = torch.cuda.FloatTensor

def to_torch(x, dtype, req = False):
  tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x

class AE(nn.Module):
  def __init__(self, n_clusters):
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
    self.code_book = nn.Parameter(torch.rand(n_clusters, 8, 2, 2).type(dtype))

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

  def load(self, loc, class_id, n_clusters):
    ae = AE(n_clusters).cuda()
    ae.load_state_dict(torch.load(loc+"_"+str(class_id)+".mdl"))
    self.ae = ae

  def torchify(self, X):
    return to_torch(X, "float").view(-1, self.n_channel, self.w_img, self.w_img)

  def learn_ae(self, X, n_clusters):
    ae = AE(n_clusters).cuda()
    for i in range(len(X) * 20):
      # load in the datas
      indices = sorted( random.sample(range(len(X)), 40) )
      X_sub = np.array([X[i] for i in indices])
      # convert to proper torch forms
      X_sub = self.torchify(X_sub)

      # optimize 
      ae.opt.zero_grad()
      
      # ------------ reconstruction loss ---------------
      output = ae(X_sub)
      loss_fun = nn.MSELoss()
      reconstruction_loss = loss_fun(output, X_sub)

      # ------------- commitment loss ------------
      enc_codes = ae.encoder(X_sub)
      # bloat up both encoded and the code_book to create all-pairs of informations
      enc_codes_bloat = enc_codes.unsqueeze(1).expand(-1, n_clusters, -1, -1, -1)
      # detach the code gradient for this one
      code_book_bloat = ae.code_book.unsqueeze(0).expand(enc_codes.size()[0], -1, -1, -1, -1)
      # all pair-wise square dist across all coordinates
      enc_code_dists = ( (code_book_bloat - enc_codes_bloat) ** 2 ).view(-1, n_clusters, 8*2*2).sum(-1)
      # print (enc_code_dists.size())
      # get the min distance
      min_dists, _argmin = enc_code_dists.min(-1)
      # print (min_dists.size())
      commitment_loss = 0.001 * torch.sum(min_dists)

      # print (reconstruction_loss, commitment_loss)
      # print (ae.code_book)
      loss = reconstruction_loss + commitment_loss

      loss.backward()
      ae.opt.step()

      if i % 4000 == 0:
        print (reconstruction_loss, commitment_loss)

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
    x = [x[0] for x in X_tsne]
    y = [x[1] for x in X_tsne]
    plt.scatter(x, y, alpha=0.5)
    plt.savefig('drawings/2d_tsne_'+str(self.class_id)+'.png')
    plt.clf()
    print (X_tsne)

  def give_clusters(self, X, n_clusters):
    X_emb = self.embed(X)
    code_book = self.ae.code_book.view(-1, 8*2*2).data.cpu().numpy()

    from sklearn.metrics import pairwise_distances_argmin_min
    # closest, _ = pairwise_distances_argmin_min(code_book, X_emb)
    closest, _ = pairwise_distances_argmin_min(X_emb, code_book)
    print (len(closest))
    print (closest[:10])
    print (X_emb[:10])
    print ("hm")
    print (code_book[18])
    print (code_book[90])
    print (code_book[85])
    assert 0, "stuf"
    counts = [cluster_labels.count(i) for i in range(n_clusters)]

    # for kk, img in enumerate(X[:10]):
    #   img = np.reshape(img, (28,28))
    #   plt.imsave('drawings/cluster_{}_{}_sample_{}.png'.format(self.class_id, cluster_labels[kk], kk), img)
    return [ (X[closest[i]], counts[i]) for i in range(n_clusters) ], kmeans.score(X_emb)

def AEnet_Maker(n_channel, w_img):
  def call(class_id):
    return AEnet(n_channel, w_img, class_id)
  return call

