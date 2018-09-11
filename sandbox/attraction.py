import torch
import numpy as np
from torch.autograd import Variable
import random
import torch.nn.functional as F

import matplotlib.pyplot as plt

dtype = torch.cuda.FloatTensor

n_points = 100
points = Variable(torch.rand(n_points, 2).type(dtype), requires_grad=True)

info = 10

for i in range(1000):
  # print (points)
  row_mat = points.unsqueeze(1).expand(-1, n_points, -1)
  col_mat = points.unsqueeze(0).expand(n_points, -1, -1)
  pair_wise_dist = ( (row_mat - col_mat) **2 ).sum(-1)
  for idxx in range(n_points):
    pair_wise_dist[idxx, idxx] = 999.0
  min_dists, _argmin = pair_wise_dist.min(-1)
  attraction_loss = torch.sum(min_dists)
  if i % info == 0:
    print (attraction_loss)
    ppp = points.data.cpu().numpy()
    x = [x[0] for x in ppp]
    y = [x[1] for x in ppp]
    plt.scatter(x, y, alpha=0.5)
    plt.savefig('pix/cluster_'+str(100 - i // info)+'.png')
    plt.clf()

  attraction_loss.backward(retain_graph=True)
  points_grad = points.grad
  points.data.sub_(points_grad.data * 0.001)
  points.grad.data.zero_()

