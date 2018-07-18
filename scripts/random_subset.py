from evaluation_models.decision_tree import DTree
from evaluation_models.logistic_regression import LRegr
from evaluation_models.svm import Svmm

from data.load_data import load_datas
import random

def get_subset(X, X_lab, K):
  indices = sorted( random.sample(range(len(X)), K) )
  X_sub = [X[i] for i in indices]
  X_lab_sub = [X_lab[i] for i in indices]
  return X_sub, X_lab_sub
  
if __name__ == "__main__":
  # load the data
  tr_img, tr_lab, t_img, t_lab = load_datas("./data/")
  # the model makers
  model_makers = [DTree, LRegr, Svmm]

  for sub_size in [100, 200, 400, 800, 1600, 3200, 6400, 12800]:
    # step 1 : get random subset
    tr_img_sub, tr_lab_sub = get_subset(tr_img, tr_lab, sub_size)
    # step 2 : make models and train them
    eval_models = [mm() for mm in model_makers]
    for mm in eval_models:
      mm.learn((tr_img_sub, tr_lab_sub))
    # step 3 : collect scores
    scores = [(mm.name, mm.evaluate((t_img, t_lab)))\
               for mm in eval_models]
    print (sub_size / len(tr_img), scores)

