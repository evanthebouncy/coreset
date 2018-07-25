from models.decision_tree import DTree
from models.logistic_regression import LRegr
from models.svm import Svmm
from models.conv_nn import Cnet_Maker
from discovery_models.random_subset import RSub

import math
from data.load_data import load_datas
  
if __name__ == "__main__":
  data_name = "cifar-10"
  # load the data
  tr_img, tr_lab, t_img, t_lab = load_datas("./data/", data_name)

  # create the model makers
  cnet_params = {
      "mnist" : (1, 28),
      "cifar-10" : (3, 32),
  }
  cnet_maker = Cnet_Maker(*cnet_params[data_name])

  # the model makers
  # model_makers = [DTree, LRegr, Svmm, cnet_maker]
  model_makers = [cnet_maker]

  # the subset selection
  r_sub = cnet_maker()
  # r_sub = RSub()

  print (" running subset selection on ", data_name, " with ", r_sub.name)

  for sub_frac in [0.0001, 0.0005, 0.001, 0.01, 0.1, 0.3, 0.5]:
    # step 1 : get random subset
    sub_size = math.floor(sub_frac * len(tr_img))
    tr_img_sub, tr_lab_sub = r_sub.get_subset(tr_img, tr_lab, sub_size)
    # step 2 : make models and train them
    eval_models = [mm() for mm in model_makers]
    for mm in eval_models:
      print ("fitting model ", mm.name)
      mm.learn((tr_img_sub, tr_lab_sub))
    # step 3 : collect scores
    scores = [(mm.name, mm.evaluate((t_img, t_lab)))\
               for mm in eval_models]
    print (sub_size / len(tr_img), scores)

