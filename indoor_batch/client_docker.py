import json
import urllib.request
import ssl
import platform
import time
import sys
import time
import pprint
import pandas as pd
import os
from time import gmtime, strftime

#'http://novasoft-th.com/indoor'
try:  
    os.environ["url"]
    url = os.environ["url"]
except KeyError: 
    print("Url is not set.")
    sys.exit(1)
    
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def bouncingSleep(retryCount):
    waittime = 8 if (retryCount[0])**2 > 8 else (retryCount[0])**2
    if((retryCount[0])**2 < 8):
        retryCount[0]+=1
    time.sleep(waittime)

def getJob():
    data = {
            'COMMAND': "REQUESTJOB",
            'NAME' : platform.node()
    }
    req = urllib.request.Request(url)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(data)
    jsondataasbytes = jsondata.encode('utf-8')

    retryCount = [0]
    while True:
        try:
            print("getting_job")#status
            response = urllib.request.urlopen(req, jsondataasbytes,context=ctx)
        except:
            print("request_error")#status
            bouncingSleep(retryCount)
            continue   
        data = json.loads(response.read().decode('utf-8'))
        if data["RESPONSE"] == "ASSIGNED":  
            print("assigned")#status
            retryCount = [0]
            return data
        elif data["RESPONSE"] == "FINISHED":
            print("finished")#status
            #sys.stdout.write('\rNo more job to do. listening for any further...')
            #sys.stdout.flush()       
            bouncingSleep(retryCount)
            continue  
        elif data["RESPONSE"] == "WAIT":
            print("wait")#status
            #sys.stdout.write('\rWait')
            #.stdout.flush()
            bouncingSleep(retryCount)
            continue         
        else:    
            bouncingSleep(retryCount)
            continue  
def submitResult(result):
    _data = {
            'COMMAND': "SUBMITRESULT",
    }
    data = {**_data , **result}
    req = urllib.request.Request(url)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(data)
    jsondataasbytes = jsondata.encode('utf-8')
    retryCount = [0]
    while True:
        try:
            response = urllib.request.urlopen(req, jsondataasbytes,context=ctx)
        except:
            retryCount += 1
            #print("Submitting failed. Retry #" + str(retry))
            print("submit_failed")#status
            bouncingSleep(retryCount)
            continue   
        data = json.loads(response.read().decode('utf-8'))
        if "RESPONSE" in data:
            retryCount = [0]
            return data
        else:
            raise Exception("unknown response")

def paramsCleanup(params):
    params["sae_hidden_layers"] =  params["sae_hidden_layers"].replace("[","")
    params["sae_hidden_layers"] =  params["sae_hidden_layers"].replace("]","")
    params["sae_hidden_layers"] =  params["sae_hidden_layers"].replace(" ","")
    params["classifier_hidden_layers"] =  params["classifier_hidden_layers"].replace("[","")
    params["classifier_hidden_layers"] =  params["classifier_hidden_layers"].replace("]","")
    params["classifier_hidden_layers"] =  params["classifier_hidden_layers"].replace(" ","")

params = getJob()
paramsCleanup(params)

#print("parameters retrived " + strftime("[%Y-%m-%d %H:%M:%S]", gmtime()))
#print ('*' * 50)
#print(pd.DataFrame(list(params.items()), columns=['Parameter', 'Value']))
#print ('*' * 50)

N = int(params["N"])
acc_bf = str(params["acc_bf"])
acc_bld = str(params["acc_bld"])
acc_flr = str(params["acc_flr"])
batch_size = int(params["batch_size"])
classifier_activation= str(params["classifier_activation"])
classifier_bias= str(params["classifier_bias"])
classifier_hidden_layers= str(params["classifier_hidden_layers"])
classifier_loss= str(params["classifier_loss"])
classifier_optimizer= str(params["classifier_optimizer"])
dropout= float(params["dropout"])
epochs= int(params["epochs"])
loc_failure= str(params["loc_failure"])
mean_pos_err= str(params["mean_pos_err"])
mean_pos_err_weighted= str(params["mean_pos_err_weighted"])
random_seed= int(params["random_seed"])
sae_activation= str(params["sae_activation"])
sae_bias= str(params["sae_bias"])
sae_hidden_layers= str(params["sae_hidden_layers"])
sae_loss= str(params["sae_loss"])
sae_optimizer= str(params["sae_optimizer"])
scaling= float(params["scaling"])
submitted_date= str(params["submitted_date"])
time_spent= int(params["time_spent"])
trained_by= str(params["trained_by"])
training_ratio= float(params["training_ratio"])

import argparse
from datetime import datetime
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

start = datetime.now()


input_dim = 520
output_dim = 13                 # number of labels
verbose = 0                     # 0 for turning off logging
INPUT_DIM = 520                 #  number of APs

path_train = 'UJIIndoorLoc/trainingData2.csv'           # '-110' for the lack of AP.
path_validation = 'UJIIndoorLoc/validationData2.csv'    # ditto


sae_hidden_layers =  [int(i) for i in (sae_hidden_layers).split(',')]
classifier_hidden_layers =  [int(i) for i in (classifier_hidden_layers).split(',')]

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

classifier_bias = str2bool(classifier_bias)
sae_bias = str2bool(sae_bias)

import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(int(random_seed))
rn.seed(12345)

import tensorflow as tf

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(random_seed)  # initialize random seed generator of tensorflow
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model

train_df = pd.read_csv(path_train, header=0) # pass header=0 to be able to replace existing names
test_df = pd.read_csv(path_validation, header=0)

train_AP_features = scale(np.asarray(train_df.iloc[:,0:520]).astype(float), axis=1)

# add a new column
train_df['REFPOINT'] = train_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1)

blds = np.unique(train_df[['BUILDINGID']])
flrs = np.unique(train_df[['FLOOR']])

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

len_train = len(train_df) 

# for consistency in one-hot encoding
blds_all = np.asarray(pd.get_dummies(pd.concat([train_df['BUILDINGID'], test_df['BUILDINGID']])))
flrs_all = np.asarray(pd.get_dummies(pd.concat([train_df['FLOOR'], test_df['FLOOR']]))) # ditto

blds = blds_all[:len_train]
flrs = flrs_all[:len_train]

rfps = np.asarray(pd.get_dummies(train_df['REFPOINT']))
train_labels = np.concatenate((blds, flrs, rfps), axis=1)
OUTPUT_DIM = train_labels.shape[1]

# split the training set into training and validation sets; 

# we will use the validation set at a testing set.
train_val_split = np.full((len(train_AP_features)), True)
train_val_split[int(len(train_AP_features)*training_ratio):len(train_AP_features)*99] = False

x_train = train_AP_features[train_val_split]
y_train = train_labels[train_val_split]
x_val = train_AP_features[~train_val_split]
y_val = train_labels[~train_val_split]

# create a model based on stacked autoencoder (SAE)
model = Sequential()
model.add(Dense(sae_hidden_layers[0], input_dim=INPUT_DIM, activation=sae_activation, use_bias=sae_bias))
for units in sae_hidden_layers[1:]:
    model.add(Dense(units, activation=sae_activation, use_bias=sae_bias))  
model.add(Dense(INPUT_DIM, activation=sae_activation, use_bias=sae_bias))
model.compile(optimizer=sae_optimizer, loss=sae_loss)

# train the model
model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=verbose,shuffle=False)

# remove the decoder part
num_to_remove = (len(sae_hidden_layers) + 1) // 2
for i in range(num_to_remove):
    model.pop()
    
### build and train a complete model with the trained SAE encoder and a new classifier
model.add(Dropout(dropout))
for units in classifier_hidden_layers:
    model.add(Dense(units, activation=classifier_activation, use_bias=classifier_bias))
    model.add(Dropout(dropout))
model.add(Dense(OUTPUT_DIM, activation='sigmoid', use_bias=classifier_bias)) # 'sigmoid' for multi-label classification
model.compile(optimizer=classifier_optimizer, loss=classifier_loss, metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=verbose,shuffle=False)

# turn the given validation set into a testing set
test_AP_features = scale(np.asarray(test_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)
x_test_utm = np.asarray(test_df['LONGITUDE'])
y_test_utm = np.asarray(test_df['LATITUDE'])
blds = blds_all[len_train:]
flrs = flrs_all[len_train:]

### evaluate the model
# calculate the accuracy of building and floor estimation
preds = model.predict(test_AP_features, batch_size=batch_size)
n_preds = preds.shape[0]

# blds_results = (np.equal(np.argmax(test_labels[:, :3], axis=1), np.argmax(preds[:, :3], axis=1))).astype(int)
blds_results = (np.equal(np.argmax(blds, axis=1), np.argmax(preds[:, :3], axis=1))).astype(int)
acc_bld = blds_results.mean()

flrs_results = (np.equal(np.argmax(flrs, axis=1), np.argmax(preds[:, 3:8], axis=1))).astype(int)
acc_flr = flrs_results.mean()
acc_bf = (blds_results*flrs_results).mean()

# calculate positioning error when building and floor are correctly estimated
mask = np.logical_and(blds_results, flrs_results) # mask index array for correct location of building and floor

x_test_utm = x_test_utm[mask]
y_test_utm = y_test_utm[mask]
blds = blds[mask]
flrs = flrs[mask]
rfps = (preds[mask])[:, 8:118]

# number of correct building and floor location
n_success = len(blds)   

n_loc_failure = 0
sum_pos_err = 0.0
sum_pos_err_weighted = 0.0
idxs = np.argpartition(rfps, -N)[:, -N:]  # (unsorted) indexes of up to N nearest neighbors
threshold = scaling*np.amax(rfps, axis=1)

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

# mean_pos_err = sum_pos_err / (n_success - n_loc_failure)
mean_pos_err = sum_pos_err / n_success
# mean_pos_err_weighted = sum_pos_err_weighted / (n_success - n_loc_failure)
mean_pos_err_weighted = sum_pos_err_weighted / n_success
loc_failure = n_loc_failure / n_success # rate of location estimation failure given that building and floor are correctly located

end = datetime.now() 
timeTaken = end - start
output = {   "N": N,
            "acc_bf": acc_bf,
            "acc_bld": acc_bld,
            "acc_flr": acc_flr,
            "batch_size": batch_size,
            "classifier_activation": classifier_activation,
            "classifier_bias": classifier_bias,
            "classifier_hidden_layers": ",".join([str(item) for item in classifier_hidden_layers]),
            "classifier_loss": classifier_loss,
            "classifier_optimizer": classifier_optimizer,
            "dropout": dropout,
            "epochs": epochs,
            "loc_failure": loc_failure,
            "mean_pos_err": mean_pos_err,
            "mean_pos_err_weighted": mean_pos_err_weighted,
            "random_seed": random_seed,
            "sae_activation": sae_activation,
            "sae_bias": sae_bias,
            "sae_hidden_layers": ",".join([str(item) for item in sae_hidden_layers]),
            "sae_loss": sae_loss,
            "sae_optimizer": sae_optimizer,
            "scaling": scaling,
            "submitted_date": datetime.now().isoformat(),
            "time_spent":  int(timeTaken.total_seconds()),
            "trained_by": platform.node(),
            "training_ratio": training_ratio
}
submit = submitResult(output)
if submit["RESPONSE"] == "SUCCESS": 
    print("submitted")#status           
    #print("result submitted " + strftime("[%Y-%m-%d %H:%M:%S]", gmtime()))
elif submit["RESPONSE"] == "FAIL":
    print("result is rejected " + strftime("[%Y-%m-%d %H:%M:%S]", gmtime()))


