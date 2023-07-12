import numpy as np

def one_class_processing(data,normal_class:int,args=None):
    labels,normal_idx,abnormal_idx=one_class_labeling(data.labels,normal_class)
    return one_class_masking(args,data,labels,normal_idx,abnormal_idx)


def one_class_labeling(labels,normal_class:int):
    normal_idx=np.where(labels==normal_class)[0]
    abnormal_idx=np.where(labels!=normal_class)[0]

    labels[normal_idx]=0
    labels[abnormal_idx]=1
    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"),normal_idx,abnormal_idx


def one_class_masking(args,data,labels,normal_idx,abnormal_idx):
    train_mask=np.zeros(labels.shape,dtype='bool')
    val_mask=np.zeros(labels.shape,dtype='bool')
    test_mask=np.zeros(labels.shape,dtype='bool')
    train_mask[normal_idx[:int(0.6*normal_idx.shape[0])]]=1
    val_mask[normal_idx[int(0.6*normal_idx.shape[0]):int(0.75*normal_idx.shape[0])]]=1
    val_mask[abnormal_idx[:int(0.15*normal_idx.shape[0])]]=1

    test_mask[normal_idx[int(0.75*normal_idx.shape[0]):]]=1
    test_mask[abnormal_idx[-int(0.25*normal_idx.shape[0]):]]=1

    return labels,train_mask,val_mask,test_mask


