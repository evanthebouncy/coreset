import torch
import numpy as np
from torch.autograd import Variable
import random
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

dtype = torch.cuda.FloatTensor

n_points = 100
n_r_subset = 20
points = Variable(torch.rand(n_points, 2).type(dtype), requires_grad=True)

weights = torch.tensor([1.0 for i in range(n_points)], dtype=torch.float)

info = 10

for i in range(10000):
  # print (points)

  r_idx = torch.multinomial(weights, n_r_subset)
  sub_points = points[r_idx]

  row_mat = sub_points.unsqueeze(0).expand(n_r_subset, -1, -1)
  col_mat = sub_points.unsqueeze(1).expand(-1, n_r_subset, -1)

  pair_wise_dist = ( (row_mat - col_mat) **2 ).sum(-1)
  for idxx in range(n_r_subset):
    pair_wise_dist[idxx, idxx] = 999.0
  min_dists, _argmin = pair_wise_dist.min(-1)
  attraction_loss = torch.sum(min_dists)
  if i % info == 0:
    print (attraction_loss)
    ppp = points.data.cpu().numpy()
    x = [x[0] for x in ppp]
    y = [x[1] for x in ppp]
    plt.scatter(x, y, alpha=0.5)
    plt.savefig('pix/k_cluster_'+str(i // info)+'.png')
    plt.clf()

  attraction_loss.backward(retain_graph=True)
  points_grad = points.grad
  points.data.sub_(points_grad.data * 0.001)
  points.grad.data.zero_()

