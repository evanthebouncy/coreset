from models.decision_tree import DTree
from models.logistic_regression import LRegr
from models.svm import Svmm
from models.conv_nn import Cnet_Maker
from models.discrete_ae import AEnet_Maker
from discovery_models.random_subset import RSub
import numpy as np
import math
from data.load_data import load_datas
import matplotlib.pyplot as plt

TRAIN = False
n_labels = 10
n_clusters = 100
  
if __name__ == "__main__":
  data_name = "mnist"
  tr_img, tr_lab, t_img, t_lab = load_datas("./data/", data_name)
  
  splits = [[] for _ in range(n_labels)]
  for idxx in range(len(tr_lab)):
    splits[tr_lab[idxx]].append(tr_img[idxx])

  # create the model makers
  ae_params = {
      "mnist" : (1, 28),
      "cifar-10" : (3, 32),
  }

  ae_maker = AEnet_Maker(*ae_params[data_name])

  # the model makers
  cnet_params = {
      "mnist" : (1, 28),
      "cifar-10" : (3, 32),
  }
  cnet_maker = Cnet_Maker(*cnet_params[data_name])
  model_makers = [cnet_maker]
  # model_makers = [DTree, LRegr, Svmm, cnet_maker]

  # the subset selection
  sub_makers = [ae_maker(class_id) for class_id in range(n_labels)]

  # train
  if TRAIN:
    print ("GOGOOGO")
    for i in range(n_labels):
      sub_makers[i].learn_ae(splits[i], n_clusters)
    for sub_maker in sub_makers:
      sub_maker.save('saved_models/discrete_ae_model')
  # don't train
  else:
    for i in range(n_labels):
      sub_makers[i].load('saved_models/discrete_ae_model', i, n_clusters)


  # for i in range(n_labels):
  #   X_emb = sub_makers[i].embed(splits[i])
  #   sub_makers[i].tsne(X_emb)

  # assert 0, "hm"

  for n_clusters in [n_clusters]:
    # representative 
    tr_img_sub = []
    tr_lab_sub = []
    cl_score = 0
    for i in range(n_labels):
      clusters, cs = sub_makers[i].give_clusters(splits[i], n_clusters)
      cl_score += cs
      for j, img_cnt in enumerate(clusters):
        img_, cnt = img_cnt
        img = np.reshape(img_, (28,28))
        plt.imsave('drawings/cluster_{}_{}.png'.format(i, j), img)

        tr_img_sub += [img_] * cnt
        tr_lab_sub += [i] * cnt
      
    tr_img_sub = np.array(tr_img_sub)
    tr_lab_sub = np.array(tr_lab_sub)

    eval_models = [mm() for mm in model_makers]
    for mm in eval_models:
      mm.learn((tr_img_sub, tr_lab_sub))
    scores = [(mm.name, mm.evaluate((t_img, t_lab)))\
               for mm in eval_models]

    # random
    sub_size = n_clusters * n_labels
    r_sub = RSub()
    r_tr_img_sub, r_tr_lab_sub = r_sub.get_subset_balanced(tr_img, tr_lab, 
                                                           sub_size, n_labels)
    # scale the size for all 60000 data, with duplicates
    bloat_factor = len(tr_lab) // len(r_tr_lab_sub)
    r_tr_img_sub = r_tr_img_sub * bloat_factor
    r_tr_lab_sub = r_tr_lab_sub * bloat_factor

    eval_models = [mm() for mm in model_makers]
    for mm in eval_models:
      mm.learn((r_tr_img_sub, r_tr_lab_sub))
    r_scores = [(mm.name, mm.evaluate((t_img, t_lab)))\
               for mm in eval_models]

    print (sub_size / len(tr_img), cl_score, scores, r_scores)




