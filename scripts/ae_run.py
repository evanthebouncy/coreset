from models.decision_tree import DTree
from models.logistic_regression import LRegr
from models.svm import Svmm
from models.plain_ae import AEnet_Maker
from discovery_models.random_subset import RSub
import numpy as np
import math
from data.load_data import load_datas

TRAIN = False
  
if __name__ == "__main__":
  data_name = "mnist"
  tr_img, tr_lab, t_img, t_lab = load_datas("./data/", data_name)
  
  splits = [[] for _ in range(10)]
  for idxx in range(len(tr_lab)):
    splits[tr_lab[idxx]].append(tr_img[idxx])

  # create the model makers
  ae_params = {
      "mnist" : (1, 28),
      "cifar-10" : (3, 32),
  }

  ae_maker = AEnet_Maker(*ae_params[data_name])

  # the model makers
  # model_makers = [DTree, LRegr, Svmm, cnet_maker]

  # the subset selection
  sub_makers = [ae_maker(class_id) for class_id in range(10)]

  # train
  if TRAIN:
    print ("GOGOOGO")
    for i in range(10):
      sub_makers[i].learn_ae(splits[i])
    for sub_maker in sub_makers:
      sub_maker.save('saved_models/ae_model')
  # don't train
  else:
    for i in range(10):
      sub_makers[i].load('saved_models/ae_model', i)

  for i in range(10):
    emb_i = sub_makers[i].embed(splits[i])
    sub_makers[i].tsne(emb_i)



