from mnist import MNIST
import numpy as np

dataset_names = ['mnist', 'cifar-10']

def load_datas(data_dir, data_name):
  assert data_name in dataset_names, "unknown data name"

  if data_name == 'mnist':
    the_path = data_dir + "/mnist/"
    mndata = MNIST(the_path)
    mndata.gz = True
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    train_images = np.array(train_images) / 255 
    test_images = np.array(test_images) / 255 
    train_labels, test_labels = np.array(train_labels), np.array(test_labels)
    return train_images, train_labels, test_images, test_labels

  if data_name == 'cifar-10':
    def unpickle(file):
      import pickle
      with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
      return dict
    root_path = data_dir + "/cifar-10/"

    train_paths = [root_path + "data_batch_"+str(i) for i in range(1,6)]
    train_dicks = [unpickle(p) for p in train_paths]
    train_datas = [d[b'data'] for d in train_dicks]
    train_datas = np.concatenate(train_datas)
    train_labels = [d[b'labels'] for d in train_dicks]
    train_labels = np.concatenate([np.array(x) for x in train_labels])

    test_path = root_path + "test_batch"
    test_dick = unpickle(test_path)
    test_datas = test_dick[b'data']
    test_labels = np.array(test_dick[b'labels'])

    return train_datas, train_labels, test_datas, test_labels

if __name__ == "__main__":
  tr_img, tr_lab, t_img, t_lab = load_datas(".", "cifar-10")
  print (len(tr_img), len(tr_lab), len(t_img), len(t_lab))
