from evaluation_models.decision_tree import DTree

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

  for sub_size in [100, 1000, 2000, 5000, 10000]:
    tr_img_sub, tr_lab_sub = get_subset(tr_img, tr_lab, sub_size)
    dtree = DTree()
    dtree.learn((tr_img_sub, tr_lab_sub))
    score = dtree.evaluate((t_img, t_lab))
    print (sub_size / len(tr_img), score)


