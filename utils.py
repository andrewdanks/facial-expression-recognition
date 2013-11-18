import scipy.io
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.ion()


def load_mat(file_name):
    return scipy.io.loadmat(file_name)

def make_submission_file(pred):
    len_pred = len(pred)
    lines = ['Id,Prediction']
    for i in range(1253):
        if i >= len_pred:
            prediction = 0
        else:
            prediction = pred[i]
        lines.append('%s,%s' % (str(i + 1), prediction))
    with open('submission.csv', 'w+') as fp:
        fp.write("\n".join(lines))

def show_metrics(actual, pred):
    print(classification_report(actual, pred, labels=range(1,8)))
    print(confusion_matrix(actual, pred, labels=range(1,8)))

def load_train():
    labeled_data = load_mat('data/labeled_data.mat')
    tr_images = labeled_data['tr_images']
    img_size, _, num_rows = tr_images.shape
    tr_images = tr_images.reshape(img_size**2, num_rows).T
    tr_labels = labeled_data['tr_labels'].reshape(-1)
    return tr_images, tr_labels

def load_valid():
    val_images = load_mat('data/val_images.mat')['val_images']
    img_size, _, num_rows = val_images.shape
    val_images = val_images.reshape(img_size**2, num_rows).T
    return val_images

def show_image(means):
  """Show the cluster centers as images."""
  plt.figure(1)
  plt.clf()
  for i in xrange(means.shape[1]):
    plt.subplot(1, means.shape[1], i+1)
    plt.imshow(means[:, i].reshape(32, 32).T, cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')