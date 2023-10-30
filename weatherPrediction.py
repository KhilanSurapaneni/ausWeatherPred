import pandas as pd
import numpy as np
import sklearn as skl 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import io
import seaborn as sns; sns.set()
from sklearn import preprocessing
plt.rc("font", size = 14)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


path = "/Users/khilansurapaneni/personalProjects/weatherPrediction/weatherAUS.csv"
data = pd.read_csv(path)
data.head(3)

#EDA
data.dropna()
data.drop('Sunshine', axis='columns', inplace=True)
data.drop('Evaporation', axis='columns', inplace=True)
data.drop('Cloud9am', axis='columns', inplace=True)
data.drop('Cloud3pm', axis='columns', inplace=True)

data = data.dropna()

#Baseline Model
# obtain all features (except target and date)
X = data.drop(['RainTomorrow','Date'],
  axis='columns')

# re-code rain today as 0 or 1 (rather than no or yes)
X.RainToday = X.RainToday.map(dict(Yes=1, No=0))

# obtain target (rain tomrrow), and encode to 0 or 1 (rather than no or yes)
Y = data['RainTomorrow']
Y = Y.map(dict(Yes=1, No=0))

#One-Hot encode all categorical variables

# Function for one-hot encoding, takes dataframe and features to encode
# returns one-hot encoded dataframe
# from: https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)

# features to one-hot encode in our data
features_to_encode = ['Location', 'WindGustDir', 'WindDir9am','WindDir3pm']

for feature in features_to_encode:
    X = encode_and_bind(X, feature)

#splitting into training and test set.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1, 1)
X_test = np.array(X_test)
y_test = np.array(y_test).reshape(-1, 1)

#linear regression
y_pred = X_test[:,12]
np.round(np.mean(y_test.reshape(1,-1)[0] == y_pred), 3)
lr = LinearRegression()
lr.fit(X_train, y_train)

train_predict_base = [1 if x>0.5 else 0 for x in lr.predict(X_train).reshape(1,-1)[0]]
test_predict_base = [1 if x>0.5 else 0 for x in lr.predict(X_test).reshape(1,-1)[0]]

print('Accuracy on train set: ', np.round(np.mean(train_predict_base == y_train.reshape(1,-1)[0]),3))
print('Accuracy on test set: ', np.round(np.mean(test_predict_base == y_test.reshape(1,-1)[0]),3))
print('\n')
print('Confusion matrix, test set (predicting rain tomorrow):')


cmtx = pd.DataFrame(
    confusion_matrix(y_test.reshape(1,-1)[0], test_predict_base, labels=[0, 1]), 
    index=['true:no', 'true:yes'], 
    columns=['pred:no', 'pred:yes']
)

#logistic model
logit_model = LogisticRegression() 
logit_model.fit(X_train, y_train)

logit_model.coef_

train_predict_base = [1 if x>0.5 else 0 for x in logit_model.predict(X_train).reshape(1,-1)[0]]
test_predict_base = [1 if x>0.5 else 0 for x in logit_model.predict(X_test).reshape(1,-1)[0]]

print('Accuracy on train set: ', np.round(np.mean(train_predict_base == y_train.reshape(1,-1)[0]),3))
print('Accuracy on test set: ', np.round(np.mean(test_predict_base == y_test.reshape(1,-1)[0]),3))
print('\n')
print('Confusion matrix, test set (predicting rain tomorrow):')

cmtx = pd.DataFrame(
    confusion_matrix(y_test.reshape(1,-1)[0], test_predict_base, labels=[0, 1]), 
    index=['true:no', 'true:yes'], 
    columns=['pred:no', 'pred:yes']
)

#neural network
X_train_nn, X_valid, y_train_nn, y_valid = train_test_split(X_train, y_train)
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train_nn) 
X_valid = scaler.transform(X_valid) 
X_test_nn = scaler.transform(X_test)


model = tf.keras.models.Sequential([
     tf.keras.layers.Dense(256, activation=tf.nn.relu), 
     tf.keras.layers.Dense(128, activation=tf.nn.relu), 
     tf.keras.layers.Dense(64, activation=tf.nn.relu), 
     tf.keras.layers.Dense(32, activation=tf.nn.relu), 
     tf.keras.layers.Dense(16, activation=tf.nn.relu), 
     tf.keras.layers.Dense(1)])

model.compile(
    loss=tf.keras.losses.binary_crossentropy, 
    optimizer= tf.keras.optimizers.legacy.Adam(), 
    metrics=['accuracy'] 
)

history = model.fit(
    X_train_nn, y_train_nn, epochs=10,
    validation_data=(X_valid, y_valid))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1]) 
plt.legend(loc='lower right') 

# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

plt.show()