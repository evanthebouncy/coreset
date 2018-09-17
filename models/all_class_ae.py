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

  # all radius are same, i.e. we only take minimum distance here
  # cluster prior is uniform
  # simple greedy strategy of max cover (with label in mind)
  def cluster_greedy(self, X, Y, n_clusters):
    n_data = len(X)
    # compute the cluster loss for a set of cluster/label against X/Y
    def cl_loss(clusters, cluster_labels, X, Y):
      k_clusters = len(clusters)
      if k_clusters == 0:
        return np.inf
      else:
        # clusters of shape [n_cluster, emb-dimension]
        # X of shape [n_data, emb-dimension]
        clusters = to_torch(clusters, "float").unsqueeze(0).expand(n_data, -1, -1)
        cluster_labels = to_torch(cluster_labels, "int")
        X = to_torch(X, "float").unsqueeze(1).expand(-1, k_clusters, -1)
        Y = to_torch(Y, "int")

        # shape of [n_data, n_clusters]
        # pair_wise_dists = ( (clusters - X) ** 2 ).sum(-1)
        pair_wise_dists =  torch.exp(-1.0 / (1.0 + torch.abs(clusters - X).sum(-1)))
        # print (pair_wise_dists)
        # shape of [n_data,] as min_dists to cluster, and [n_data,] as argmin
        min_dists, argmin = pair_wise_dists.min(-1)
        # [n_data, n_cluster]
        cluster_label_bloat = cluster_labels.unsqueeze(0).expand(n_data, -1)
        label_pred = cluster_label_bloat.gather(1, argmin.view(-1, 1)).view(-1)
        mistakes = label_pred != Y
        return torch.sum(mistakes).data.cpu().numpy()

    def greedy_1step(prev_clusters, prev_labels):
      best_cls, best_labs, best_val = None, None, np.inf
      for i in range(n_data):
        x, y = X[i], Y[i]
        cand_clusters = list(prev_clusters) + [x]
        cand_labels   = list(prev_labels) + [y]

        cand_clusters, cand_labels = np.array(cand_clusters), np.array(cand_labels)
        loss = cl_loss(cand_clusters, cand_labels, X, Y)
        if loss < best_val:
          best_cls, best_labs, best_val = cand_clusters, cand_labels, loss

      return best_cls, best_labs, best_val

    cls, labs, val = [], [], 999
    for step in range(n_clusters):
      cls, labs, val = greedy_1step(cls, labs)
      print (val)
    print (labs)
    assert 0, "asdf"
        

    print( cl_loss([], X) )

  def give_clusters(self, X, n_clusters):
    X_emb = self.embed(X)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans = kmeans.fit(X_emb)


    cluster_labels = list(kmeans.predict(X_emb))
    from sklearn.metrics import pairwise_distances_argmin_min
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_emb)
    counts = [cluster_labels.count(i) for i in range(n_clusters)]

    # for kk, img in enumerate(X[:10]):
    #   img = np.reshape(img, (28,28))
    #   plt.imsave('drawings/cluster_{}_{}_sample_{}.png'.format(self.class_id, cluster_labels[kk], kk), img)
    return [ (X[closest[i]], counts[i]) for i in range(n_clusters) ], kmeans.score(X_emb)

  def make_knn(self, X, Y):
    X_emb = self.embed(X)
    from sklearn import neighbors
    clf = neighbors.KNeighborsClassifier(1)
    knn_cl = clf.fit(X_emb, Y)
    def classify(X_new):
      X_new_emb = self.embed(X_new)
      return knn_cl.predict(X_new_emb)
    return classify
    


def AEnet_Maker(n_channel, w_img):
  def call():
    return AEnet(n_channel, w_img)
  return call




