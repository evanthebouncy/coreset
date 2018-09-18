from models.decision_tree import DTree
from models.logistic_regression import LRegr
from models.svm import Svmm
from models.conv_nn import Cnet_Maker
from models.all_class_ae import AEnet_Maker
from discovery_models.random_subset import RSub
import numpy as np
import math
from data.load_data import load_datas
import matplotlib.pyplot as plt

TRAIN = False
n_labels = 10
n_samples = 100

def evaluate_subset(t_img, t_lab, img_sub, lab_sub, model_makers):
  eval_models = [mm() for mm in model_makers]
  for mm in eval_models:
    mm.learn((img_sub, lab_sub))
  scores = [(mm.name, mm.evaluate((t_img, t_lab)))\
             for mm in eval_models]
  print (scores)
  
if __name__ == "__main__":
  data_name = "mnist"
  tr_img, tr_lab, t_img, t_lab = load_datas("./data/", data_name)
  
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
  sub_maker = ae_maker()

  # train
  if TRAIN:
    print ("GOGOOGO")
    sub_maker.learn_ae(tr_img)
    sub_maker.save('saved_models/all_class_ae_model')
  # don't train
  else:
    sub_maker.load('saved_models/all_class_ae_model')

    r_idxs = np.random.choice(np.arange(len(tr_img)), n_samples, replace=False)
    img_sub_r = tr_img[r_idxs] 
    lab_sub_r = tr_lab[r_idxs]
    evaluate_subset(t_img, t_lab, img_sub_r, lab_sub_r, model_makers)

    img_sub_p, lab_sub_p = sub_maker.sub_select2(n_samples, tr_img, tr_lab,
        embed=True, inc_size = 1)
    evaluate_subset(t_img, t_lab, img_sub_p, lab_sub_p, model_makers)



