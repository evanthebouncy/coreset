from mnist import MNIST

def load_datas(the_path):
  mndata = MNIST(the_path)
  mndata.gz = True
  train_images, train_labels = mndata.load_training()
  test_images, test_labels = mndata.load_testing()
  return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
  tr_img, tr_lab, t_img, t_lab = load_datas(".")
  print (len(tr_img), len(tr_lab), len(t_img), len(t_lab))
