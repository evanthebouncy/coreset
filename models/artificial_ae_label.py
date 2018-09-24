import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random
from data.artificial_data1 import render_pic
import matplotlib.pyplot as plt

def knn_loss(clf, X, Y):
  pred = clf(X)
  incorrect = (pred != Y)
  return sum(incorrect)

class AE(nn.Module):

  def __init__(self):
    super(AE, self).__init__()
    n_hidden_big = 30
    n_hidden = 16
    # for encding
    self.enc_fc1 = nn.Linear(100, n_hidden_big)
    self.enc_fc2 = nn.Linear(n_hidden_big, n_hidden)
    # for decoding
    self.dec_fc1 = nn.Linear(n_hidden, n_hidden_big)
    self.dec_fc2 = nn.Linear(n_hidden_big, 100)
    # for predicting
    self.pred_fc1 = nn.Linear(n_hidden, n_hidden)
    self.pred_fc2 = nn.Linear(n_hidden, 1)

    self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

  def enc(self, x):
    '''
    the x input here is encoded as batch x channel x L x L
    '''
    x = x.view(-1, 100)
    x = F.relu(self.enc_fc1(x))
    x = F.sigmoid(self.enc_fc2(x))
    return x

  def dec(self, x):
    x = F.relu(self.dec_fc1(x))
    x = F.sigmoid(self.dec_fc2(x))
    x = x.view(-1, 10, 10)
    return x

  def predict(self, x):
    x = F.relu(self.pred_fc1(x))
    x = F.sigmoid(self.pred_fc2(x))
    x = x.view(-1)
    return x

  def forward(self, x):
    x = self.enc(x)
    x = self.dec(x)
    return x

  def xentropy_loss(self, target, prediction):
    prediction = torch.clamp(prediction, 1e-5, 1.0-1e-5)
    target_prob1 = target
    target_prob0 = 1.0 - target 
    log_prediction_prob1 = torch.log(prediction)
    log_prediction_prob0 = torch.log(1.0 - prediction)

    loss_value = -torch.sum(target_prob1 * log_prediction_prob1 + \
                            target_prob0 * log_prediction_prob0)
    return loss_value

class AEnet():
  # takes in channel variable n and width of image
  def __init__(self):
    self.name = "AEnet"

  def save(self, loc):
    torch.save(self.ae.state_dict(), loc+".mdl")

  def load(self, loc):
    ae = AE().cuda()
    ae.load_state_dict(torch.load(loc+".mdl"))
    self.ae = ae
    return ae

  def torchify(self, x):
    x = Variable(torch.from_numpy(x).type(torch.cuda.FloatTensor), 
        requires_grad = False)
    return x

  def learn_ae(self, X, Y):
    ae = AE().cuda()
    for i in range(len(X) * 2):
      # load in the datas
      indices = sorted( random.sample(range(len(X)), 40) )
      X_sub = self.torchify(X[indices])
      Y_sub = self.torchify(Y[indices])

      X_emb = ae.enc(X_sub)
      X_rec = ae.dec(X_emb)
      Y_pred = ae.predict(X_emb)
      reconstruct_loss = ae.xentropy_loss(X_sub, X_rec) * 0.1
      prediction_loss =  ae.xentropy_loss(Y_sub, Y_pred)

      loss = reconstruct_loss # + prediction_loss

      # optimize 
      ae.opt.zero_grad()
      loss.backward()
      ae.opt.step()

      if i % 4000 == 0:
        print (i, "rec loss ", reconstruct_loss, "classify loss ", prediction_loss)

        X_sub_np =  X_sub[0].data.cpu().view(10,10).numpy()
        output_np = X_rec[0].data.cpu().view(10,10).numpy()
        render_pic(X_sub_np, 'drawings/art_sam.png')
        render_pic(output_np, 'drawings/art_rec.png')

    self.ae = ae
    return ae

  def embed(self, X):
    X = np.array(X)
    X = self.torchify(X)
    encoded = self.ae.enc(X)
    return encoded.data.cpu().numpy()

  def tsne(self, embedded_X, labels=None, name=None):
    cl_colors = np.linspace(0, 1, len(labels)) if (labels is not None) else ['blue']
    from sklearn.manifold import TSNE
    X_tsne = TSNE(n_components=2).fit_transform(embedded_X)
    import matplotlib
    # matplotlib.use("svg")
    x = [x[0] for x in X_tsne]
    y = [x[1] for x in X_tsne]
    colors = [cl_colors[lab] for lab in labels] if (labels is not None) else 'blue'
    plt.scatter(x, y, c=colors, alpha=0.5)
    name = name if name else ""
    plt.savefig('drawings/2d_tsne_ae_artificial'+name+'.png')
    plt.clf()
    return X_tsne

  def make_knn(self, X, Y, embed=True):
    X_emb = self.embed(X) if embed else X
    from sklearn import neighbors
    # I HRD SOMEWHERE THAT USING k = LOG(len(DATA)) is good
    clf = neighbors.KNeighborsClassifier(1+int(np.log(len(Y))))
    # clf = neighbors.KNeighborsClassifier(1+int(np.log(len(Y))), weights='distance')
    knn_cl = clf.fit(X_emb, Y)
    def classify(X_new):
      X_new_emb = self.embed(X_new) if embed else X_new
      return knn_cl.predict(X_new_emb)
    return classify

  def sub_select2(self, n_samples, X, Y, embed=True, inc_size=10, search_width=10):
    def loss(X_sub, Y_sub):
      clf = self.make_knn(X_sub, Y_sub, embed)
      return knn_loss(clf, X, Y)

    # get a batch of new samples from remaining set
    def make_sample(X_rem, Y_rem, inc_size):
      r_idxs = np.random.choice(np.arange(len(X_rem)), inc_size, replace=False)
      return X_rem[r_idxs], Y_rem[r_idxs]

    def one_step(X_sub, Y_sub, inc_size, search_width):
      samples = [make_sample(X, Y, inc_size) for _ in range(search_width)]
      cand_sub = [(np.concatenate((X_sub, samp[0])), np.concatenate((Y_sub, samp[1]))) for samp in samples] if len(X_sub) > 0 else samples

      loss_cand = [(loss(*cand), cand) for cand in cand_sub]
      best_score, best_cand = min(loss_cand, key = lambda t: t[0])
      print (best_score)
      return best_cand

    X_sub, Y_sub = [], []
    for iter_n in range(n_samples // inc_size):
      print (iter_n)
      X_sub, Y_sub = one_step(X_sub, Y_sub, inc_size, search_width) 

    return X_sub, Y_sub

  def sub_select4(self, n_samples, X, Y, embed=True, inc_size=10, search_width=10):
    def loss(X_sub, Y_sub):
      clf = self.make_knn(X_sub, Y_sub, embed)
      return knn_loss(clf, X, Y)

    # get a batch of new samples from remaining set
    def make_sample(X_rem, Y_rem, inc_size):
      if len(X_rem) < inc_size:
        return X_rem, Y_rem
      r_idxs = np.random.choice(np.arange(len(X_rem)), inc_size, replace=False)
      return X_rem[r_idxs], Y_rem[r_idxs]

    def get_rem(X_sub, Y_sub):
      if X_sub == []:
        return X, Y
      else:
        clf = self.make_knn(X_sub, Y_sub, embed)
        pred = clf(X)
        incorrect = (pred != Y)
        X_remain = X[incorrect]
        Y_remain = Y[incorrect]
        return X_remain, Y_remain

    def one_step(X_sub, Y_sub, inc_size, search_width):
      
      X_rem, Y_rem = get_rem(X_sub, Y_sub)
      samples = [make_sample(X_rem, Y_rem, inc_size) for _ in range(search_width)]
      cand_sub = [(np.concatenate((X_sub, samp[0])), np.concatenate((Y_sub, samp[1]))) for samp in samples] if len(X_sub) > 0 else samples
      if len(X_sub) > 0:
        cand_sub.append((X_sub, Y_sub))

      loss_cand = [(loss(*cand), cand) for cand in cand_sub]
      best_score, best_cand = min(loss_cand, key = lambda t: t[0])
      print (best_score)
      return best_cand

    X_sub, Y_sub = [], []
    for iter_n in range(n_samples // inc_size):
      print (iter_n)
      X_sub, Y_sub = one_step(X_sub, Y_sub, inc_size, search_width) 

    return X_sub, Y_sub


def AEnet_Maker():
  def call():
    return AEnet()
  return call

if __name__ == '__main__':
  print ("hi")
  import pickle
  LOC = "./data/artificial1/artificial1.p"
  X,Y = pickle.load(open(LOC,"rb"))
  X_tr, Y_tr = X[:60000], Y[:60000]

  aenet = AEnet_Maker()()
  aenet.learn_ae(X_tr, Y_tr)
  aenet.save('saved_models/artificial_ae_model')
