import numpy as np
import matplotlib.pyplot as plt

L = 10

# takes in parameters: center x, center y, width
# returns a predicate function that draws the square
def mk_square(s_x,s_y,w):
  def square(x,y):
    return s_x-w<=x<=s_x+w and s_y-w<=y<=s_y+w
  return square

def mk_cross(c_x,c_y):
  def cross(x,y):
    return int(c_x) == x or int(c_y) == y
  return cross

def add_noise(pred):
  def noise(x,y):
    result = pred(x,y)
    if np.random.random() < 0.1:
      return False
    else:
      return result
  return noise

def gen_data1():
  n_square = 0
  n_cross = 0
  # generate a list of shapes 
  shapes = []
  for i in range(2):
    x = np.random.random() * L
    y = np.random.random() * L
    toss = np.random.random()
    if toss > 0.5:
      shapes.append( mk_square(x,y,2) )
      n_square += 1
    else:
      shapes.append( mk_cross(x,y) )
      n_cross += 1
  ret = np.zeros((L,L))
  for i in range(L):
    for j in range(L):
      ret[i][j] = 1.0 if max([shape(i, j) for shape in shapes]) else 0.0

  return ret, n_square == n_cross
# return ret, n_square > n_cross

def gen_data2():
  n_square = 0
  n_cross = 0
  # generate a list of shapes 
  shapes = []
  for i in range(1):
    x = np.random.random() * L
    y = np.random.random() * L
    toss = np.random.random()
    if toss > 0.5:
      shapes.append( add_noise(mk_square(x, y, 2) ))
      n_square += 1
    else:
      shapes.append( add_noise(mk_cross(x, y) ))
      n_cross += 1
  ret = np.zeros((L,L))
  for i in range(L):
    for j in range(L):
      ret[i][j] = 1.0 if max([shape(i, j) for shape in shapes]) else 0.0

  return ret, n_square > 0

def gen_data(kind):
  gens = {
      1 : gen_data1,
      2 : gen_data2,
      }
  return gens[kind]()
  

def gen_data_set(n_data, kind):
  datas = [gen_data(kind) for _ in range(n_data)]
  X = np.array([d[0] for d in datas])
  Y = np.array([1 if d[1] else 0 for d in datas])
  return X, Y

FIG = plt.figure()
def render_pic(m, name):
  FIG.clf()

  matrix = m
  orig_shape = np.shape(matrix)
  new_shape = orig_shape
  matrix = np.reshape(matrix, new_shape)
  ax = FIG.add_subplot(1,1,1)
  ax.set_aspect('equal')
  plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
  # plt.colorbar()
  plt.savefig(name)

if __name__ == '__main__':
  kind = 2
  # check 1 step generation is right
  shape, exist = gen_data(kind)
  render_pic(shape, 'data/artificial1/square_cross_sample.png')
  print (shape)
  print(exist)

  get_data = True
  if get_data:
    X,Y = gen_data_set(70000, kind)
    import pickle

    LOC = "./data/artificial1/artificial1.p"
    pickle.dump((X,Y), open(LOC, "wb"))
    XX,YY = pickle.load(open(LOC,"rb"))
    print (XX,YY)

