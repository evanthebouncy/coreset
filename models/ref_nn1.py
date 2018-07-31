from sklearn import svm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models.utils import to_torch 
import random

class RefNN(nn.Module):
  '''
    reference neural network contains 3 parts : 
     - qry-maker takes in a single un-labeled image and produce 2 keys, k1 and k2
     - ref-maker takes in a set of reference image-label pairs and produce k-v pairs
     - all the references k are matched against k1 and k2 in a dot-product and
       their v summed to form ctx1 and ctx2
     - predictor takes in the ctx1 and ctx2 to produce a final output prediction label
  '''
  def __init__(self, n_channel, w_img, n_labels):
    super(RefNN, self).__init__()
    # ============= Constants ============
    self.n_img_latent = 40
    self.n_kv_latent  = 40
    self.n_qry  = 20
    self.n_val = 20

    # ============= Encoding an Image ============
    # some shannanigans to find out the n-hidden (i don't know the right function lul)
    conv_hiddens = {
          (1, 28) : 320,
          (3, 32) : 500,
        }
    self.conv_hidden = conv_hiddens[(n_channel, w_img)] 
    # 1 input channel, 10 output channel, 5x5 sliding window
    self.conv1 = nn.Conv2d(n_channel, 10, 5)
    self.conv2 = nn.Conv2d(10, 20, 5)
    self.fc1 = nn.Linear(self.conv_hidden, self.n_img_latent)

    # =========== GENERATING 2 KEYS ============
    self.qry = nn.Linear(self.n_img_latent, self.n_img_latent)
    self.mk_qry1 = nn.Linear(self.n_img_latent, self.n_qry)
    self.mk_qry2 = nn.Linear(self.n_img_latent, self.n_qry)

    # =========== Generate K-V PAIRS ============
    self.kv = nn.Linear(self.n_img_latent + n_labels, self.n_kv_latent)
    self.mk_key = nn.Linear(self.n_kv_latent, self.n_qry)
    self.mk_val = nn.Linear(self.n_kv_latent, self.n_val)

    # ========== Make the Prediction ============
    self.pred = nn.Linear(self.n_val + self.n_val, n_labels)
    
    # ========== Optimization ============
    self.opt = torch.optim.Adam(self.parameters(), lr=0.0002)

  # takes in an image and encode into n_img_latent
  def enc_img(self, img):
    x = img
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = x.view(-1, self.conv_hidden)
    x = F.relu(self.fc1(x))
    return x
  
  # takes in n_img_latent and produce 2 keys of n_qry
  def gen_qrys(self, img_lat):
    x = F.relu(self.qry(img_lat))
    qry1 = self.mk_qry1(x)
    qry2 = self.mk_qry2(x)
    return qry1, qry2

  # takes in n_img_latent and a label and produce k-v pair of n_qry n_key
  def gen_kv(self, ref_img_lats, ref_labs):
    '''
      ref_img_lats : [ M x batch x n_img_latent ]
      ref_labs     : [ M x batch x n_labels ]
    '''
    x = torch.cat((ref_img_lats, ref_labs), dim=2)
    x = F.relu(self.kv(x))
    key = self.mk_key(x)
    val = self.mk_val(x)
    return key, val

  # takes in a query, M keys, create weights
  def make_W(self, qry, keys):
    '''
      qry :  [ batch x n_qry ]
      keys : [ M x batch x n_qry ]
    '''
    M = keys.size()[0]
    qry = qry.unsqueeze(0)
    qry = qry.expand(M, -1, -1) # match the key dimension [M x batch x n_qry]
    W = torch.sum(qry * keys, dim=2) # [ M x batch ] 
    W = W  / W.norm(2, 0, keepdim=True) # [ M x batch ] normalized along axis 0
    # W = F.softmax(F.tanh(W), dim=0)
    return W

  # takes in a qry, M keys and M values, produce a context based on the qry
  def make_ctx(self, qry, keys, vals):
    '''
      qry :  [ batch x n_qry ]
      keys : [ M x batch x n_qry ]
      vals : [ M x batch x n_val ]
    '''
    W = self.make_W(qry, keys) # [ M x batch ]
    W = W.unsqueeze(-1) # [ M x batch x 1 ]
    W = W.expand(-1, -1, self.n_val) # [ M x batch x n_val ]
    weighted_vals = W * vals # [ M x batch x n_val ]
    ctx = torch.sum(weighted_vals, dim = 0) # [ batch x n_val ]
    '''
    pray to rnJESUS this is right
    '''
    return ctx
    
  # takes in 2 context and make the prediction 
  def make_pred(self, ctx1, ctx2):
    ctx = torch.cat((ctx1, ctx2), dim=1)
    logits = self.pred(ctx)
    return F.log_softmax(logits, dim=1) 

  # takes in an image, a set of reference image and labels, output a prediction
  def forward(self, img, ref_imgs, ref_labs):
    '''
      img :      [ batch x channel x W x W ]
      ref_imgs : [ M x batch x channel x W x W ] 
      ref_labs : [ M x batch x n_lab ]
    '''
    img_lat = self.enc_img(img)
    qry1, qry2 = self.gen_qrys(img_lat)

    # list of M of [batch x chan x W x W]
    ref_img_list = torch.unbind(ref_imgs)
    # list of M of [batch x n_img_lat]
    ref_img_lat_list = [self.enc_img(ref_img) for ref_img in ref_img_list]
    ref_img_lats = torch.stack(ref_img_lat_list) # [ M x batch x n_img_latent ]
    keys, vals = self.gen_kv(ref_img_lats, ref_labs)

    ctx1, ctx2 = self.make_ctx(qry1, keys, vals), self.make_ctx(qry2, keys, vals)
    pred = self.make_pred(ctx1, ctx2)
    return pred

  def get_ref_W(self, img, ref_imgs, ref_labs):
    '''
      img :      [ batch x channel x W x W ]
      ref_imgs : [ M x batch x channel x W x W ] 
      ref_labs : [ M x batch x n_lab ]
    '''
    img_lat = self.enc_img(img)
    qry1, qry2 = self.gen_qrys(img_lat)

    # list of M of [batch x chan x W x W]
    ref_img_list = torch.unbind(ref_imgs)
    # list of M of [batch x n_img_lat]
    ref_img_lat_list = [self.enc_img(ref_img) for ref_img in ref_img_list]
    ref_img_lats = torch.stack(ref_img_lat_list) # [ M x batch x n_img_latent ]
    keys, vals = self.gen_kv(ref_img_lats, ref_labs)

    W1, W2 = self.make_W(qry1, keys), self.make_W(qry2, keys)
    return W1, W2


def generate_ref_data(imgs, labs):
  # use the rsub for random subset for now
  r_sub = RSub()

  # generate the image references and label references
  ref_imgs = []
  ref_labs = []
  for i in range(100):
    img_batch, lab_batch = r_sub.get_subset(imgs, labs, 40)
    ref_imgs.append(img_batch)
    # convert the label to 1-hot here
    lab_batch = np.array(lab_batch)
    lab_batch = np.eye(10)[lab_batch]
    ref_labs.append(lab_batch)
  ref_imgs = np.array(ref_imgs)
  ref_labs = np.array(ref_labs)

  ref_imgs = to_torch(ref_imgs, "float").view(100, -1, 1, 28, 28)
  ref_labs = to_torch(ref_labs, "float").view(100, -1, 10)

  # generate img input
  img_batch, lab_batch = r_sub.get_subset(imgs, labs, 40)
  img_batch = to_torch(np.array(img_batch), "float").view(-1, 1, 28, 28)
  lab_batch = to_torch(np.array(lab_batch), "int")

  return img_batch, ref_imgs, ref_labs, lab_batch


if __name__ == "__main__":
  print ("hello we're about to crash")
  ref_nn = RefNN(1, 28, 10).cuda()

  from data.load_data import load_datas
  from discovery_models.random_subset import RSub

  data_name = "mnist"
  # load the data
  tr_img, tr_lab, t_img, t_lab = load_datas("./data/", data_name)

  for _ in range(1000):
    # generate training batch
    img_batch, ref_imgs, ref_labs, lab_batch = generate_ref_data(tr_img, tr_lab)
    # optimize weights
    ref_nn.opt.zero_grad()
    pred = ref_nn(img_batch, ref_imgs, ref_labs)
    loss = F.nll_loss(pred, lab_batch)
    loss.backward()
    ref_nn.opt.step()
    print (loss)

    # print some diagnostics
    if _ % 100 == 0:
      W1, W2 = ref_nn.get_ref_W(img_batch, ref_imgs, ref_labs)
      W1_batch0 = W1.data.cpu().numpy().transpose()[0]
      W2_batch0 = W2.data.cpu().numpy().transpose()[0]
      print (W1_batch0)



























