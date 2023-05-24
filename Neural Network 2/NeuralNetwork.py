# Tensorflow/keras
from tensorflow import keras
from keras.models import Sequential # for making linear stack of layers for neural network
from keras import Input # for making Keras tensor
from keras.layers import Dense # for making regular densely-connected NN layers

# Data manipulation
import pandas as pd
import numpy as np

# Sklearn
import sklearn # for model evaluation
from sklearn.model_selection import train_test_split # split data into samples
from sklearn.metrics import classification_report # model evaluation metrics

# Visualization
import plotly
import plotly.express as px
import plotly.graph_objects as go

# Set Pandas options to display more columns
pd.options.display.max_columns=50

# Read in the weather data csv
df=pd.read_csv('weatherAUS.csv', encoding='utf-8')

# Drop records where target RainTomorrow=NaN
df=df[pd.isnull(df['RainTomorrow'])==False]

#kept getting errors bc the input data might have multiple types in a column(in some rows), so forcing types
df = df.apply(pd.to_numeric, errors='coerce')


# For other columns with missing values, fill them in with column mean
df=df.fillna(df.mean())

# Create a flag for RainToday and RainTomorrow, note RainTomorrowFlag will be our target variable
df['RainTodayFlag']=df['RainToday'].apply(lambda x: 1 if x=='Yes' else 0)
df['RainTomorrowFlag']=df['RainTomorrow'].apply(lambda x: 1 if x=='Yes' else 0)

# Show a snapshot of data
df


#Now, time to make a ff neural network based on one variable

#Step 1 - select data
X = df[['Humidity3pm']]
y = df['RainTomorrowFlag'].values

#Step 2 - Make training and testing samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Step 3 - describe neural network structure
model = Sequential(name="Model-with-One-Input") #model
model.add(Input(shape=(1,), name='Input-Layer')) #add input layer, specify shape of inputs
model.add(Dense(2, activation='softplus', name='Hidden-Layer')) #add hidden layer with the softplus function: log(exp(x)+1)
model.add(Dense(1, activation='sigmoid', name='Output-Layer')) #another hidden layer with sigmoid function: 1 / (1 + exp(-x))

#Step 4 - compile Keras model
model.compile(optimizer='adam', #'rmsprop' by default, backpropagation algorithm
              loss = 'binary_crossentropy', #specifying loss function(here, it's binary crossentropy)(string or tf.keras.losses.Loss instance)
              metrics=['Accuracy', 'Precision', 'Recall'], # metrics used in training/testing, string name of function or tf.keras.metrics.Metric instance
              loss_weights=None, # default=none, optional list of float scalars to weight loss
              weighted_metrics=None, #default=none, list of metrics evaluated and weighted using sample_weight or class_weight in training/testing
              run_eagerly=None, #default=false. If true, model's logic not wrapped in tf function. Recommended to stay none(unless cannot use tf)
              steps_per_execution=None) #default=1, # of batches per single tf function call(improves performance on TPUs and small models)

#Step 5 - Fit keras model on dataset
model.fit(X_train, #input data
          y_train, #target data
          batch_size=10, #number of samples per gradient update(default=32)
          epochs=3, #number of epochs, iteration over input and test data(default=1)
          verbose='auto', #default=auto(auto, 0, 1, 2), verbosity mode, 0=silent, 1=progress bar, 2=one line per epoch, auto is 1 unless ParameterServerStrategy(2)
          callbacks=None, #default=none, list of callbacks during training
          validation_split=0.2, #default=0.0, percentage of data to be used for validation(evaluate loss and model metrics on this data at end of each epoch, without training on it)
          #validation_data=(X_test, y_test), #default=none, data for validation
          shuffle=True, #default=True, whether to shuffle data before each epoch
          class_weight={0: 0.3, 1 : 0.7}, #default=none, optional dictionary mapping class indices to a weight(weighting loss function during training. (pay more attention to some things at beginning)
          sample_weight=None, #default=none, optional numpy array of weights for training samples, weighting loss function during training
          initial_epoch=0, #default=0, which epoch to start at(useful for resuming prev)
          steps_per_epoch=None, #default=None(can be integer), number of batches before declaring end of epoch. When tensorflow data tensors, none=#samples in dataset / batchsize, or 1 if dk
          validation_steps=None, #number of batches to look at for validation at end of each epoch(only matters if have validation_data)
          validation_batch_size=None, #default=None(can be integer), #samples per validation batch, none=batch_size
          validation_freq=3, #default=1, can be None, how many epochs to run before new validation run
          max_queue_size=10, #default=10, used for generator/keras.util.Sequence input, max size of generator queue
          workers=1, #default=1, used for generator/keras..., max #processes to spin when process-based threading
          use_multiprocessing=False) #default=false, used for gen/ker..., if true, process-based threading

#Step 6 - Use model to make predictions
#predict class labels on training data
pred_labels_tr = (model.predict(X_train) > 0.5).astype(int)
#predict class labels on test data
pred_labels_te = (model.predict(X_test) > 0.5).astype(int)

#Step 7 - Model performance summary
print("")
print('-------------------- Model Summary --------------------')
model.summary() # print model summary
print("")
print('-------------------- Weights and Biases --------------------')
for layer in model.layers:
    print("Layer: ", layer.name)
    print("  --Kernels(weights): ", layer.get_weights()[0])#weights
    print("  --Biases: ", layer.get_weights()[1])#biases

print("")
print('---------- Evaluation on Training Data ----------')
print(classification_report(y_train, pred_labels_tr))
print("")

print('---------- Evaluation on Test Data ----------')
print(classification_report(y_test, pred_labels_te))
print("")




