from evaluation_models.decision_tree import DTree
from evaluation_models.logistic_regression import LRegr
from evaluation_models.svm import Svmm
from evaluation_models.conv_nn import Cnet

from discovery_models.logistic_active import LRegrActive

from data.load_data import load_datas
  
if __name__ == "__main__":
  # load the data
  tr_img, tr_lab, t_img, t_lab = load_datas("./data/")
  # the model makers
  # model_makers = [DTree, LRegr, Svmm]
  model_makers = [Cnet]
  # the subset selection
  r_sub = LRegrActive("min_prob")

  for sub_size in [100, 200, 400, 800, 1600, 3200, 6400, 12800]:
    # step 1 : get random subset
    tr_img_sub, tr_lab_sub = r_sub.get_subset(tr_img, tr_lab, sub_size)
    print ("subset size : ", sub_size / len(tr_img))
    # step 2 : make models and train them
    eval_models = [mm() for mm in model_makers]
    for mm in eval_models:
      print (" fitting model ", mm.name)
      mm.learn((tr_img_sub, tr_lab_sub))
    # step 3 : collect scores
    scores = [(mm.name, mm.evaluate((t_img, t_lab)))\
               for mm in eval_models]
    print ("scores : ", scores)

