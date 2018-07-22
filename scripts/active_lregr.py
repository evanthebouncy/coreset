from evaluation_models.decision_tree import DTree
from evaluation_models.logistic_regression import LRegr
from evaluation_models.svm import Svmm
from evaluation_models.conv_nn import Cnet_Maker

from discovery_models.logistic_active import LRegrActive

from data.load_data import load_datas
  
if __name__ == "__main__":
  # data_name = "cifar-10"
  data_name = "mnist"
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
  r_sub = LRegrActive("min_prob")

  for sub_size in [100, 200, 400, 800, 1600, 3200, 6400, 12800]:
    # step 1 : get random subset
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

