#@title Imports 

# Import libraries for data processing.
import numpy as np
import pandas as pd
import warnings
import time
warnings.filterwarnings('ignore')

# Import Tensorflow and Keras Libraries.
import tensorflow as tf
from tensorflow.python.keras import backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.models import Model
from keras.layers import GlobalAveragePooling2D,MaxPooling2D
from keras.layers import Dense,Flatten,SpatialDropout2D
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import History

# Import sklearn libraries for data processing.
import sklearn
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Install and import Pycaret library for transformation and classification.
from pycaret.classification import *



df = pd.read_excel('Data_Insurance_TGI.xlsx')
df.ffill(axis = 0,inplace=True)

numFeatures = 'Accident/Health_(P)_Duration', 'Accident/Health_(P)_Amount', 'Builders_Risk_(P)_Duration', 'Builders_Risk_(P)_Amount', 'Dwelling_Fire_Duration', 'Dwelling_Fire_Amount', 'Earthquake_(P)_Duration', 'Earthquake_(P)_Amount', 'Flood_Duration', 'Flood_Amount', 'Homeowners_Duration', 'Homeowners_Amount', 'Life_(P)_Duration', 'Life_(P)_Amount', 'Motorcycle_Duration', 'Motorcycle_Amount', 'Private_Passenger_Auto_Duration', 'Private_Passenger_Auto_Amount', 'Umbrella_(P)_Duration', 'Umbrella_(P)_Amount'

customer_class = setup(data=df,
                       target='StillCustomer',
                       session_id=786,
                       transformation=True,
                       normalize=False,
                       train_size=0.95,
                       numeric_features=(numFeatures),
                       remove_outliers=False,
                       outliers_threshold=0.05,
                       remove_multicollinearity=False,
                       multicollinearity_threshold=0.9,
                       feature_selection=False,
                       bin_numeric_features=['DurationAsCust'],
                       feature_interaction=False,
                       silent=True,
                       ignore_features=['Customer_ID', 'BecameCust'])

X_train = customer_class[2]
X_test = customer_class[3]
y_train = customer_class[4]
y_test = customer_class[5]

model = tf.keras.models.load_model('/bestModel.hdf5')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=128, epochs=5)

y_pred = model.predict_classes(X_test)
y_pred = model.predict_classes(X_test)
matrix = metrics.confusion_matrix(y_test, y_pred)
print(matrix)
