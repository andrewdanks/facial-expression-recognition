import scipy.io
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def load_mat(file_name):
    return scipy.io.loadmat(file_name)

def make_submission_file(pred, file_name='submission.csv'):
    len_pred = len(pred)
    lines = ['Id,Prediction']
    for i in range(1253):
        if i >= len_pred:
            prediction = 0
        else:
            prediction = pred[i]
        lines.append('%s,%s' % (str(i + 1), prediction))
    with open(file_name, 'w+') as fp:
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

def load_unlabelled():
    unlabeled_images = load_mat('data/unlabeled_images.mat')['unlabeled_images']
    img_size, _, num_rows = unlabeled_images.shape
    unlabeled_images = unlabeled_images.reshape(img_size**2, num_rows).T
    return unlabeled_images

def generate_arff(X, y, file_name='data'):
    lines = ['@RELATION fr']
    num_data, num_dim = X.shape
    for i in range(num_dim):
        lines.append('@ATTRIBUTE d%s NUMERIC' % i)
    lines.append('@ATTRIBUTE class {1,2,3,4,5,6,7}')
    lines.append('@DATA')
    for i in range(num_data):
        x = []
        for j in range(num_dim):
            x.append(X[i][j])
        x.append(y[i])
        lines.append(','.join(map(str, x)))
    with open(file_name + '.arff', 'w+') as fp:
        fp.write("\n".join(lines))
