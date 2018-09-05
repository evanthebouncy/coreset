from sklearn import svm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models.utils import to_torch 
import random

# nearest neighbor neural network
class NNNN(nn.Module):
  '''
    reference neural network contains 3 parts : 
     - encoder that takes in an image and encode it into a lower dimensional space
     - all the images in the ref image set are scored based on the dot-product distance 
     - the weighted sum of label is added to produce the final label
  '''
  def __init__(self, n_channel, w_img, n_labels):
    super(NNNN, self).__init__()
    # ============= Constants ============
    self.n_img_latent = 40
    self.n_labels = n_labels

    # ============= Encoding an Image ============
    # some shannanigans to find out the n-hidden (i don't know the right function lul)
    conv_hiddens = {
          (1, 28) : 320,
          (3, 32) : 500,
        }
    self.n_conv_hidden = conv_hiddens[(n_channel, w_img)] 
    # 1 input channel, 10 output channel, 5x5 sliding window (don't change these LOL)
    self.conv1 = nn.Conv2d(n_channel, 10, 5)
    self.conv2 = nn.Conv2d(10, 20, 5)
    self.fc1 = nn.Linear(self.n_conv_hidden, self.n_img_latent)

    # ========== Optimization ============
    self.opt = torch.optim.Adam(self.parameters(), lr=0.01)

  # takes in an image and encode into n_img_latent
  def enc_img(self, img):
    x = img
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = x.view(-1, self.n_conv_hidden)
    x = F.sigmoid(self.fc1(x))
    return x
  
  # takes in a query, M keys, create weights
  def make_W(self, qry, keys):
    '''
      qry :  [ batch x n_qry ]
      keys : [ M x batch x n_qry ]
    '''
    M = keys.size()[0]
    qry = qry.unsqueeze(0) # [1 x batch x n_qry]
    qry = qry.expand(M, -1, -1) # match the key dimension [M x batch x n_qry]
    W = torch.sum(qry * keys, dim=2) # [ M x batch ] 
    W = F.softmax(W, dim=0) # [M x batch]
    return W
    
  # takes in an image, a set of reference image and labels, output a prediction
  def forward(self, img, ref_imgs, ref_labs):
    '''
      img :      [ batch x channel x W x W ]
      ref_imgs : [ M x batch x channel x W x W ] 
      ref_labs : [ M x batch x n_lab ]
    '''
    qry = self.enc_img(img)

    # list of M of [batch x chan x W x W]
    ref_img_list = torch.unbind(ref_imgs)
    # list of M of [batch x n_img_lat]
    ref_img_lat_list = [self.enc_img(ref_img) for ref_img in ref_img_list]
    ref_keys = torch.stack(ref_img_lat_list) # [ M x batch x n_img_latent ]

    W = self.make_W(qry, ref_keys) # [M x batch]
    W = W.unsqueeze(-1) # [ M x batch x 1 ]
    W = W.expand(-1, -1, self.n_labels) # [ M x batch x n_labels ]
    weighted_labels = W * ref_labs # [ M x batch x n_val ]
    pred = torch.sum(weighted_labels, dim = 0) + 0.001 # [ batch x n_val ]
    log_pred = torch.log(pred)
    return log_pred

  def get_ref_W(self, img, ref_imgs, ref_labs):
    '''
      img :      [ batch x channel x W x W ]
      ref_imgs : [ M x batch x channel x W x W ] 
      ref_labs : [ M x batch x n_lab ]
    '''
    qry = self.enc_img(img)

    # list of M of [batch x chan x W x W]
    ref_img_list = torch.unbind(ref_imgs)
    # list of M of [batch x n_img_lat]
    ref_img_lat_list = [self.enc_img(ref_img) for ref_img in ref_img_list]
    ref_keys = torch.stack(ref_img_lat_list) # [ M x batch x n_img_latent ]

    W = self.make_W(qry, ref_keys) # [M x batch]
    return W


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
  print ("hello we're about to E X P L O D ")
  ref_nn = NNNN(1, 28, 10).cuda()

  from data.load_data import load_datas
  from discovery_models.random_subset import RSub

  data_name = "mnist"
  # load the data
  tr_img, tr_lab, t_img, t_lab = load_datas("./data/", data_name)

  for _ in range(10000):
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
      W = ref_nn.get_ref_W(img_batch, ref_imgs, ref_labs)
      W_batch0 = W.data.cpu().numpy().transpose()[0]
      print (W_batch0)

