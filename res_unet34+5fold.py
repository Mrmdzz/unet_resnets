import os
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")



import cv2
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook #, tnrange
#from itertools import chain
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
#from skimage.morphology import label

from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,ZeroPadding2D
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose,UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler
from keras import backend as K
from keras import optimizers

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

import time
t_start = time.time()

cv_total = 5
#cv_index = 1 -5


version = 56
basic_name_ori = f'Unet_resnet_v{version}'
save_model_name_0= basic_name_ori + '.model'
submission_file = basic_name_ori + '.csv'

print(save_model_name_0)
print(submission_file)

img_size_ori = 101
img_size_target = 101

def upsample(img):# not used
    return img
    
def downsample(img):# not used
    return img
	
# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss	
LR = 0.002
NB_EPOCHS = 100
NB_SNAPSHOTS = 5
def cosine_anneal_schedule(epoch):
    
    cos_inner = np.pi * (epoch % (NB_EPOCHS // NB_SNAPSHOTS))
    cos_inner /= NB_EPOCHS // NB_SNAPSHOTS
    cos_outer = np.cos(cos_inner) + 1
    lrs=float(LR / 2 * cos_outer)
    if lrs<0.001:
        return 0.001
    else :
        return lrs
# Loading of training/testing ids and depths
train_df = pd.read_csv("E:/kaggleTGS/RawData/all/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("E:/kaggleTGS/RawData/all/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)


train_df["images"] = [np.array(load_img("E:/kaggleTGS/RawData/all/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]


train_df["masks"] = [np.array(load_img("E:/kaggleTGS/RawData/all/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]

#### Reference  from Heng's discussion
# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63984#382657
def get_mask_type(mask):
    border = 10
    outer = np.zeros((101-2*border, 101-2*border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border, borderType = cv2.BORDER_CONSTANT, value = 1)

    cover = (mask>0.5).sum()
    if cover < 8:
        return 0 # empty
    if cover == ((mask*outer) > 0.5).sum():
        return 1 #border
    if np.all(mask==mask[0]):
        return 2 #vertical

    percentage = cover/(101*101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7

def histcoverage(coverage):
    histall = np.zeros((1,8))
    for c in coverage:
        histall[0,c] += 1
    return histall

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_target, 2)

train_df["coverage_class"] = train_df.masks.map(get_mask_type)

train_all = []
evaluate_all = []
skf = StratifiedKFold(n_splits=cv_total, random_state=1234, shuffle=True)
for train_index, evaluate_index in skf.split(train_df.index.values, train_df.coverage_class):
    train_all.append(train_index)
    evaluate_all.append(evaluate_index)
    print(train_index.shape,evaluate_index.shape) # the shape is slightly different in different cv, it's OK
	
	
def get_cv_data(cv_index):
    train_index = train_all[cv_index-1]
    evaluate_index = evaluate_all[cv_index-1]
    x_train = np.array(train_df.images[train_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_train = np.array(train_df.masks[train_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    x_valid = np.array(train_df.images[evaluate_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_valid = np.array(train_df.masks[evaluate_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    return x_train,y_train,x_valid,y_valid
	
	
cv_index = 1
train_index = train_all[cv_index-1]
evaluate_index = evaluate_all[cv_index-1]

print(train_index.shape,evaluate_index.shape)
histall = histcoverage(train_df.coverage_class[train_index].values)
print(f'train cv{cv_index}, number of each mask class = \n \t{histall}')
histall_test = histcoverage(train_df.coverage_class[evaluate_index].values)
print(f'evaluate cv{cv_index}, number of each mask class = \n \t {histall_test}')

fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(24, 6), sharex=True, sharey=True)

# show mask class example
for c in range(8):
    j= 0
    for i in train_index:
        if train_df.coverage_class[i] == c:
            axes[j,c].imshow(np.array(train_df.masks[i])  )
            axes[j,c].set_axis_off()
            axes[j,c].set_title(f'class {c}')
            j += 1
            if(j>=2):
                break
				
				
def BatchActivate(x):
    #x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Lambda(lambda x: K.elu(x) + 1)(x)
    return x
def residual_block(blockInput,num, DropoutRatio,start_neurons=16):

    x = Conv2D(start_neurons * num, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(blockInput) #,kernel_initializer='he_normal'
    x = BatchActivate(x)

    x = Conv2D(start_neurons * num, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(x)
    x = BatchActivate(x)
    #x = Dropout(DropoutRatio)(x)

    x = Conv2D(start_neurons * num, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(x)
    x = BatchActivate(x)
    #x = Dropout(DropoutRatio)(x)

    x = Conv2D(start_neurons * num, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(x)
    x = BatchActivate(x)

    x = Add()([x, blockInput])
    #x = BatchNormalization()(x)
    x = BatchActivate(x)
    return x
#11

def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(input_layer)
    conv1 = BatchActivate(conv1)
    conv1 = residual_block(conv1,1,DropoutRatio/2)
    conv1 = residual_block(conv1,1,DropoutRatio/2)   
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(pool1)
    conv2 = BatchActivate(conv2)
    conv2 = residual_block(conv2,2,DropoutRatio)
    conv2 = residual_block(conv2,2,DropoutRatio)   
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(pool2)
    conv3 = BatchActivate(conv3)
    conv3 = residual_block(conv3,4,DropoutRatio)
    conv3 = residual_block(conv3,4,DropoutRatio)   
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(pool3)
    conv4 = BatchActivate(conv4)
    conv4 = residual_block(conv4,8,DropoutRatio)
    conv4 = residual_block(conv4,8,DropoutRatio)   
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(pool4)
    convm = BatchActivate(convm)
    convm = residual_block(convm,16,DropoutRatio)
    convm = residual_block(convm,16,DropoutRatio) 
    
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(convm)
    #deconv4 = Conv2D(start_neurons * 8, 2, activation=None, padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(convm))
    deconv4= BatchActivate(deconv4)	
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = BatchNormalization()(uconv4)
    uconv4 = Dropout(DropoutRatio)(uconv4)    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(uconv4)
    uconv4 = BatchActivate(uconv4)
    uconv4 = residual_block(uconv4,8,DropoutRatio)
    uconv4 = residual_block(uconv4,8,DropoutRatio) 
    
    
    # 12 -> 25
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid",kernel_initializer='he_normal')(uconv4)
    #deconv3 = Conv2D(start_neurons * 4, 2, activation=None, padding='valid',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(uconv4))
    #deconv3=ZeroPadding2D(padding=(1, 1))(deconv3)
    deconv3= BatchActivate(deconv3)	
    uconv3 = concatenate([deconv3, conv3]) 
    uconv3 = BatchNormalization()(uconv3)	
    uconv3 = Dropout(DropoutRatio)(uconv3)    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(uconv3)
    uconv3 = BatchActivate(uconv3)
    uconv3 = residual_block(uconv3,4,DropoutRatio)
    uconv3 = residual_block(uconv3,4,DropoutRatio) 
    

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same",kernel_initializer='he_normal')(uconv3)
    #deconv2 = Conv2D(start_neurons * 2, 2, activation=None, padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(uconv3))
    deconv2= BatchActivate(deconv2)
    uconv2 = concatenate([deconv2, conv2]) 
    uconv2 = BatchNormalization()(uconv2)	
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(uconv2)
    uconv2 = BatchActivate(uconv2)
    uconv2 = residual_block(uconv2,2,DropoutRatio)
    uconv2 = residual_block(uconv2,2,DropoutRatio) 
    
    
    # 50 -> 101
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid",kernel_initializer='he_normal')(uconv2)
    #deconv1 = Conv2D(start_neurons * 1, 2, activation=None, padding='valid',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(uconv2))
    #deconv1=ZeroPadding2D(padding=(1, 1))(deconv1)
    deconv1= BatchActivate(deconv1)
    uconv1 = concatenate([deconv1, conv1]) 
    uconv1 = BatchNormalization()(uconv1)	
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchActivate(uconv1)
    uconv1 = residual_block(uconv1,1,DropoutRatio)
    uconv1 = residual_block(uconv1,1,DropoutRatio) 
    
    
    uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None,kernel_initializer='he_normal')(uconv1)
    #output_layer_noActi = BatchNormalization()(output_layer_noActi)
    output_layer =  Activation('sigmoid')(output_layer_noActi)
    
    return output_layer#229
	
def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
#             metric.append(1)
#             continue
        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)
def get_iou_vector_1(A, B):
    A = np.squeeze(A) # new added 
    B = np.squeeze(B) # new added
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue
        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0)  )/ (np.sum(union > 0) )
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)
def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred >0], tf.float64)

def build_complie_modelone(lr = 0.01):
    input_layer = Input((img_size_target, img_size_target, 1))
    output_layer = build_model(input_layer, 16,0.5)

    model1 = Model(input_layer, output_layer)

    c = optimizers.adam(lr = lr)
    model1.compile(loss="weighted_bce_dice_loss", optimizer=c, metrics=[my_iou_metric])#binary_crossentropy weighted_bce_dice_loss
    return model1
def build_complie_modeltwo(lr = 0.01):
    input_layer = Input((img_size_target, img_size_target, 1))
    output_layer = build_model(input_layer, 16,0.5)

    model1 = Model(input_layer, output_layer)

    c = optimizers.adam(lr = lr)
    model1.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])
    return model1
	
def plot_history(history,metric_name):
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_score.plot(history.epoch, history.history[metric_name], label="Train score")
    ax_score.plot(history.epoch, history.history["val_" + metric_name], label="Validation score")
    ax_score.legend()

def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in


    #true_objects = 2
    #pred_objects = 2

    #  if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
#     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
    #print(temp1)
    intersection = temp1[0]
    #print("temp2 = ",temp1[1])
    #print(intersection.shape)
   # print(intersection)
    # Compute areas (needed for finding the union between all objects)
    #print(np.histogram(labels, bins = true_objects))
    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    #print("area_true = ",area_true)
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
  
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)  	
# training
ious = [0] * cv_total
## Scoring for last model, choose threshold by validation data 
thresholds_ori = np.linspace(0.4, 0.5, 50)
# Reverse sigmoid function: Use code below because the  sigmoid activation was removed
thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

for cv_index in range(cv_total):
    basic_name = f'Unet_resnet_v{version}_cv{cv_index+1}'
    print('############################################\n', basic_name)
    save_model_name = basic_name + '.model'
    
    x_train, y_train, x_valid, y_valid =  get_cv_data(cv_index+1)
    
    #Data augmentation
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

    """model = build_complie_modelone(lr = 0.01)

    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',
                                  factor=0.5, patience=5, min_lr=0.001, verbose=1)

    epochs = 40 #small number for demonstration 
    batch_size = 32
    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[ model_checkpoint,reduce_lr], 
                        verbose=2)
    #plot_history(history,'my_iou_metric')
    
    #model.load_weights(save_model_name)"""
    model1 = load_model(save_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,'lovasz_loss': lovasz_loss})
    input_x = model1.layers[0].input

    output_layer = model1.layers[335].output
    model = Model(input_x, output_layer)
    #model.summary()
    #c = optimizers.Adam(lr=0.001)
    c=optimizers.Adam(lr=0.002)#注意,待定!!!
    model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])
    #model = build_complie_modeltwo(lr = 0.01)

    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2', 
                                   mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',
                                  factor=0.5, patience=30, min_lr=0.0005, verbose=1)
    lr_schedule = LearningRateScheduler(schedule=cosine_anneal_schedule, verbose=1)

    epochs = 100#small number for demonstration 
    batch_size = 32
    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[ model_checkpoint,lr_schedule], 
                        verbose=2)
    plot_history(history,'my_iou_metric_2')
    
    model=load_model(save_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,'lovasz_loss': lovasz_loss})
    
    preds_valid = predict_result(model,x_valid,img_size_target)
    ious[cv_index] = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
  
threshold_best_index=[]
for cv_index in range(cv_total):
    print(f"cv {cv_index} ious = {ious[cv_index]}")
    threshold_best_index.append(np.argmax(ious[cv_index]))#寻找每个阀值列表的最大值对应的下标，组成一个列表
    print(np.argmax(ious[cv_index]))
Max=0.0
Max_index=0  
print(threshold_best_index) 
for each in range(len(threshold_best_index)):#找到这个由最大值组成的列表中的最大值
    if ious[each][threshold_best_index[each]]>Max:
        Max=ious[each][threshold_best_index[each]]
        Max_index=each
        
print(Max_index)
print(Max)	
iou_best = Max
threshold_best = thresholds[Max_index]	
print(threshold_best)
"""
used for converting the decoded image to rle mask
Fast compared to previous one
"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
	
x_test = np.array([(np.array(load_img("E:/kaggleTGS/RawData/all/test/images/{}.png".format(idx), grayscale = True))) / 255 for idx in tqdm_notebook(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)

# average the predictions from different folds
t1 = time.time()
preds_test = np.zeros(np.squeeze(x_test).shape)
for cv_index in range(cv_total):
    basic_name = f'Unet_resnet_v{version}_cv{cv_index+1}'
    model=load_model(basic_name + '.model',custom_objects={'my_iou_metric_2': my_iou_metric_2,'lovasz_loss': lovasz_loss})
    preds_test += predict_result(model,x_test,img_size_target) /cv_total
    
t2 = time.time()
print(f"Usedtime = {t2-t1} s")

t1 = time.time()
#threshold_best  = 0.5 # some value in range 0.4- 0.5 may be better 
pred_dict = {idx: rle_encode(np.round(preds_test[i]) > threshold_best) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
t2 = time.time()

print(f"Usedtime = {t2-t1} s")

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(submission_file)
t_finish = time.time()
print(f"Kernel run time = {(t_finish-t_start)/3600} hours")
