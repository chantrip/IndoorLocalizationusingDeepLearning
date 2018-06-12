
# coding: utf-8

# In[247]:


import argparse
import datetime
import os
import math
import numpy as np
import pandas as pd
import sys
from keras.utils import plot_model
from sklearn.preprocessing import scale
from timeit import default_timer as timer
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot



# In[248]:


### set paramter values
#------------------------------------------------------------------------
# general
#------------------------------------------------------------------------
training_ratio = 0.9            # ratio of training data to overall data
input_dim = 520
output_dim = 13                 # number of labels
verbose = 1                     # 0 for turning off logging
seed = 7                        # random number seed for reproducibility
### global constant variables
#------------------------------------------------------------------------
# general
#------------------------------------------------------------------------
INPUT_DIM = 520                 #  number of APs
VERBOSE = 1                     # 0 for turning off logging
#------------------------------------------------------------------------
# stacked auto encoder (sae)
#------------------------------------------------------------------------
# SAE_ACTIVATION = 'tanh'
SAE_ACTIVATION = 'relu'
SAE_BIAS = False
SAE_OPTIMIZER = 'adam'
SAE_LOSS = 'mse'
#------------------------------------------------------------------------
# classifier
#------------------------------------------------------------------------
CLASSIFIER_ACTIVATION = 'relu'
CLASSIFIER_BIAS = False
CLASSIFIER_OPTIMIZER = 'adam'
CLASSIFIER_LOSS = 'binary_crossentropy'
#------------------------------------------------------------------------
# input files
#------------------------------------------------------------------------
path_train = '../data/UJIIndoorLoc/trainingData2.csv'           # '-110' for the lack of AP.
path_validation = '../data/UJIIndoorLoc/validationData2.csv'    # ditto
#------------------------------------------------------------------------
# output files
#------------------------------------------------------------------------
path_base = '../my_results/'
path_out =  path_base + ''
#path_sae_model = path_base + '_sae_model.hdf5'

batch_size = 10
epochs = 20
#sae_hidden_layers = [256,128,64,128,256]
sae_hidden_layers = [64, 64, 256, 512]
#classifier_hidden_layers = [128,128]
classifier_hidden_layers =  [64,512,128]
dropout = 0.2
N = 8
scaling= 0.2

random_seed = 0


# In[249]:


### initialize random seed generator of numpy
#import os
#os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(random_seed)


# In[250]:


import tensorflow as tf

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,device_count = {'GPU': 0})
#session_conf = tf.ConfigProto(device_count = {'GPU': 0})
from keras import backend as K
tf.set_random_seed(random_seed)  # initialize random seed generator of tensorflow
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model


# In[251]:


train_df = pd.read_csv(path_train, header=0) # pass header=0 to be able to replace existing names
test_df = pd.read_csv(path_validation, header=0)


# In[252]:


train_AP_features = scale(np.asarray(train_df.iloc[:,0:520]).astype(float), axis=1)


# In[253]:


train_AP_features


# In[254]:


# add a new column
train_df['REFPOINT'] = train_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1)


# In[255]:


blds = np.unique(train_df[['BUILDINGID']])
flrs = np.unique(train_df[['FLOOR']])


# In[256]:


x_avg = {}
y_avg = {}
for bld in blds:
    for flr in flrs:
        # map reference points to sequential IDs per building-floor before building labels
        cond = (train_df['BUILDINGID']==bld) & (train_df['FLOOR']==flr)
        
        _, idx = np.unique(train_df.loc[cond, 'REFPOINT'], return_inverse=True) # refer to numpy.unique manual
        train_df.loc[cond, 'REFPOINT'] = idx
            
        # calculate the average coordinates of each building/floor
        x_avg[str(bld) + '-' + str(flr)] = np.mean(train_df.loc[cond, 'LONGITUDE'])
        y_avg[str(bld) + '-' + str(flr)] = np.mean(train_df.loc[cond, 'LATITUDE'])


# In[257]:


len_train = len(train_df) 
len_train


# In[258]:


# for consistency in one-hot encoding
blds_all = np.asarray(pd.get_dummies(pd.concat([train_df['BUILDINGID'], test_df['BUILDINGID']])))
flrs_all = np.asarray(pd.get_dummies(pd.concat([train_df['FLOOR'], test_df['FLOOR']]))) # ditto

blds = blds_all[:len_train]
flrs = flrs_all[:len_train]


# In[259]:


rfps = np.asarray(pd.get_dummies(train_df['REFPOINT']))
train_labels = np.concatenate((blds, flrs, rfps), axis=1)
OUTPUT_DIM = train_labels.shape[1]


# In[260]:


# split the training set into training and validation sets; 

# we will use the validation set at a testing set.
train_val_split = np.full((len(train_AP_features)), True)
train_val_split[int(len(train_AP_features)*training_ratio):len(train_AP_features)] = False


# In[261]:


np.sum(train_val_split)


# In[262]:



x_train = train_AP_features[train_val_split]
y_train = train_labels[train_val_split]
x_val = train_AP_features[~train_val_split]
y_val = train_labels[~train_val_split]


# In[263]:


# create a model based on stacked autoencoder (SAE)
model = Sequential()
model.add(Dense(sae_hidden_layers[0], input_dim=INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
for units in sae_hidden_layers[1:]:
    model.add(Dense(units, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))  
model.add(Dense(INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS)

# train the model
#model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=VERBOSE,shuffle=False)
model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=VERBOSE)




# In[264]:


# remove the decoder part
num_to_remove = (len(sae_hidden_layers) + 1) // 2
for i in range(num_to_remove):
    model.pop()
    
### build and train a complete model with the trained SAE encoder and a new classifier
model.add(Dropout(dropout))
for units in classifier_hidden_layers:
    model.add(Dense(units, activation=CLASSIFIER_ACTIVATION, use_bias=CLASSIFIER_BIAS))
    model.add(Dropout(dropout))
model.add(Dense(OUTPUT_DIM, activation='sigmoid', use_bias=CLASSIFIER_BIAS)) # 'sigmoid' for multi-label classification
model.compile(optimizer=CLASSIFIER_OPTIMIZER, loss=CLASSIFIER_LOSS, metrics=['accuracy'])


# In[265]:


model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=VERBOSE)


# In[266]:


# turn the given validation set into a testing set
test_AP_features = scale(np.asarray(test_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)
x_test_utm = np.asarray(test_df['LONGITUDE'])
y_test_utm = np.asarray(test_df['LATITUDE'])
blds = blds_all[len_train:]
flrs = flrs_all[len_train:]


# In[267]:


### evaluate the model
# calculate the accuracy of building and floor estimation
preds = model.predict(test_AP_features, batch_size=batch_size)
n_preds = preds.shape[0]

np.savetxt('building_score_cpu_4.csv',preds[:, :3], delimiter=',')
preds[:, :3]


# In[268]:


# blds_results = (np.equal(np.argmax(test_labels[:, :3], axis=1), np.argmax(preds[:, :3], axis=1))).astype(int)
blds_results = (np.equal(np.argmax(blds, axis=1), np.argmax(preds[:, :3], axis=1))).astype(int)
acc_bld = blds_results.mean()


# In[269]:


np.argmax(blds, axis=1)


# In[270]:


flrs_results = (np.equal(np.argmax(flrs, axis=1), np.argmax(preds[:, 3:8], axis=1))).astype(int)
acc_flr = flrs_results.mean()
acc_bf = (blds_results*flrs_results).mean()


# In[271]:


acc_bf


# In[272]:


# calculate positioning error when building and floor are correctly estimated
mask = np.logical_and(blds_results, flrs_results) # mask index array for correct location of building and floor


# In[273]:


x_test_utm = x_test_utm[mask]
y_test_utm = y_test_utm[mask]
blds = blds[mask]
flrs = flrs[mask]
rfps = (preds[mask])[:, 8:118]


# In[274]:


preds


# In[275]:


# number of correct building and floor location
n_success = len(blds)   
n_success


# In[276]:


n_loc_failure = 0
sum_pos_err = 0.0
sum_pos_err_weighted = 0.0
idxs = np.argpartition(rfps, -N)[:, -N:]  # (unsorted) indexes of up to N nearest neighbors
threshold = scaling*np.amax(rfps, axis=1)


# In[277]:


idxs


# In[278]:


np.argpartition(rfps, -N)


# In[279]:


np.argpartition(rfps, -N)[:, -N:]


# In[280]:


rfps.shape


# In[281]:


for i in range(n_success):
    xs = []
    ys = []
    ws = []
    for j in idxs[i]:
        rfp = np.zeros(110)
        rfp[j] = 1
        rows = np.where((train_labels == np.concatenate((blds[i], flrs[i], rfp))).all(axis=1)) # tuple of row indexes
        if rows[0].size > 0:
            if rfps[i][j] >= threshold[i]:
                xs.append(train_df.loc[train_df.index[rows[0][0]], 'LONGITUDE'])
                ys.append(train_df.loc[train_df.index[rows[0][0]], 'LATITUDE'])
                ws.append(rfps[i][j])
    if len(xs) > 0:
        sum_pos_err += math.sqrt((np.mean(xs)-x_test_utm[i])**2 + (np.mean(ys)-y_test_utm[i])**2)
        sum_pos_err_weighted += math.sqrt((np.average(xs, weights=ws)-x_test_utm[i])**2 + (np.average(ys, weights=ws)-y_test_utm[i])**2)
    else:
        n_loc_failure += 1
        key = str(np.argmax(blds[i])) + '-' + str(np.argmax(flrs[i]))
        pos_err = math.sqrt((x_avg[key]-x_test_utm[i])**2 + (y_avg[key]-y_test_utm[i])**2)
        sum_pos_err += pos_err
        sum_pos_err_weighted += pos_err


# In[282]:


# mean_pos_err = sum_pos_err / (n_success - n_loc_failure)
mean_pos_err = sum_pos_err / n_success
# mean_pos_err_weighted = sum_pos_err_weighted / (n_success - n_loc_failure)
mean_pos_err_weighted = sum_pos_err_weighted / n_success
loc_failure = n_loc_failure / n_success # rate of location estimation failure given that building and floor are correctly located


# In[283]:


### print out final results
now = datetime.datetime.now
path_out = "../my_results/"
#path_out += "_" + now.strftime("%Y%m%d-%H%M%S") + ".txt"
path_out += "[SAE" + str(sae_hidden_layers) + "] "
path_out += "[Class" +str(classifier_hidden_layers) +  "] "
path_out += "[DropOut" + str(dropout)+"] "
path_out += "[PE" + str(round(mean_pos_err,2))+"] "
path_out += "[PEW" + str(round(mean_pos_err_weighted,2))+"] "
path_out += ".txt"
f = open(path_out, 'w')
f.write("#+STARTUP: showall\n")  # unfold everything when opening
f.write("* System parameters\n")
f.write("  - Numpy random number seed: %d\n" % random_seed)
f.write("  - Ratio of training data to overall data: %.2f\n" % training_ratio)
f.write("  - Number of epochs: %d\n" % epochs)
f.write("  - Batch size: %d\n" % batch_size)
f.write("  - Number of neighbours: %d\n" % N)
f.write("  - Scaling factor for threshold: %.2f\n" % scaling)
f.write("  - SAE hidden layers: %d" % sae_hidden_layers[0])
for units in sae_hidden_layers[1:]:
    f.write("-%d" % units)
f.write("\n")
f.write("  - SAE activation: %s\n" % SAE_ACTIVATION)
f.write("  - SAE bias: %s\n" % SAE_BIAS)
f.write("  - SAE optimizer: %s\n" % SAE_OPTIMIZER)
f.write("  - SAE loss: %s\n" % SAE_LOSS)
f.write("  - Classifier hidden layers: ")
if classifier_hidden_layers == '':
    f.write("N/A\n")
else:
    f.write("%d" % classifier_hidden_layers[0])
    for units in classifier_hidden_layers[1:]:
        f.write("-%d" % units)
    f.write("\n")
f.write("  - Classifier hidden layer activation: %s\n" % CLASSIFIER_ACTIVATION)
f.write("  - Classifier bias: %s\n" % CLASSIFIER_BIAS)
f.write("  - Classifier optimizer: %s\n" % CLASSIFIER_OPTIMIZER)
f.write("  - Classifier loss: %s\n" % CLASSIFIER_LOSS)
f.write("  - Classifier dropout rate: %.2f\n" % dropout)
# f.write("  - Classifier class weight for buildings: %.2f\n" % building_weight)
# f.write("  - Classifier class weight for floors: %.2f\n" % floor_weight)
f.write("* Performance\n")
f.write("  - Accuracy (building): %e\n" % acc_bld)
f.write("  - Accuracy (floor): %e\n" % acc_flr)
f.write("  - Accuracy (building-floor): %e\n" % acc_bf)
f.write("  - Location estimation failure rate (given the correct building/floor): %e\n" % loc_failure)
f.write("  - Positioning error (meter): %e\n" % mean_pos_err)
f.write("  - Positioning error (weighted; meter): %e\n" % mean_pos_err_weighted)
f.close()


# In[284]:


mean_pos_err_weighted


# In[285]:


#9.277102567499114
#9.826676547695087
#9.836135151319597
#9.875814741893349


# In[286]:


#9.516800349169966
#9.516800349169966
#9.516800349169966
#9.516800349169966
#9.516800349169966


# In[287]:


#9.402408856112912
#9.402408856112912
#9.402408856112912
#8.714254619707846
#8.714254619707846

