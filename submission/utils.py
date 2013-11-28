import scipy.io

def load_mat(file_name):
    return scipy.io.loadmat(file_name)

def load_train():
    labeled_data = load_mat('labeled_data.mat')
    tr_images = fix_dimensions(labeled_data['tr_images'])
    tr_labels = labeled_data['tr_labels'].reshape(-1)
    return tr_images, tr_labels

def load_valid():
    return fix_dimensions(load_mat('val_images.mat')['val_images'])

def load_test():
    return fix_dimensions(load_mat('test_images.mat')['test_images'])

def load_unlabelled():
    return fix_dimensions(load_mat('unlabeled_images.mat')['unlabeled_images'])

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