from models.decision_tree import DTree
from models.logistic_regression import LRegr
from models.svm import Svmm
from models.conv_nn import Cnet_Maker
from models.all_class_artae import AEnet_Maker
from discovery_models.random_subset import RSub
import numpy as np
import math
from data.load_data import load_datas
import matplotlib.pyplot as plt

TRAIN = False
n_samples = 100

def evaluate_subset(t_img, t_lab, img_sub, lab_sub, model_makers):
  # print (t_img.shape)
  # t_img, img_sub = np.reshape(t_img, (-1, 28*28)), np.reshape(img_sub, (-1, 28*28))
  # print (t_img.shape)
  eval_models = [mm() for mm in model_makers]
  for mm in eval_models:
    mm.learn((img_sub, lab_sub))
  scores = [(mm.name, mm.evaluate((t_img, t_lab)))\
             for mm in eval_models]
  print (scores)
  
if __name__ == "__main__":
  import pickle
  LOC = "./data/artificial1/artificial1.p"                                              
  X,Y = pickle.load(open(LOC,"rb"))                                                     
  X_tr, Y_tr = X[:60000], Y[:60000]                                                     
  X_t, Y_t = X[60000:], Y[60000:]

  # the model makers
  cnet_maker = Cnet_Maker(1,28,2)
  model_makers = [cnet_maker]

  # quick tsne check
  X_tsne, Y_tsne = X_tr[:1000], Y_tr[:1000]
  ae_maker = AEnet_Maker(1, 28)
  tsne_maker = ae_maker()
  tsne_maker.load('saved_models/artificial_allae_model')
  X_tsne_emb = tsne_maker.embed(X_tsne)
  tsne_maker.tsne(X_tsne_emb, Y_tsne)

  sub_maker = AEnet_Maker(1, 28)()
  sub_maker.load('saved_models/artificial_allae_model')

  r_idxs = np.random.choice(np.arange(len(X_tr)), n_samples, replace=False)
  X_sub_r = X_tr[r_idxs] 
  Y_sub_r = Y_tr[r_idxs]
  evaluate_subset(X_t, Y_t, X_sub_r, Y_sub_r, model_makers)

  X_sub_p, Y_sub_p = sub_maker.sub_select2(n_samples, X_tr, Y_tr,
      embed=True, inc_size = 2)
  evaluate_subset(X_t, Y_t, X_sub_p, Y_sub_p, model_makers)



