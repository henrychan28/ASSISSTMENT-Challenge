
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os
import sys


# In[31]:


data_dir = './data/'

student_log_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('student_log')]
training_label_path = os.path.join(data_dir, 'training_label.csv')
validation_test_label = os.path.join(data_dir, 'validation_test_label.csv')

dfs = []
for path in student_log_paths:
    temp = pd.read_csv(path)
    dfs.append(temp)
student_df = pd.concat(dfs)

training_label_df = pd.read_csv(training_label_path)
validation_test_label_df = pd.read_csv(validation_test_label)


# In[32]:


print("student_df.shape:", student_df.shape) 
print("training_label_df.shape:", training_label_df.shape)
print("validation_test_label_df.shape:", validation_test_label_df.shape)


# In[33]:


student_specific_columns = ["AveKnow",
                            "AveCarelessness",
                            "AveCorrect",
                            "NumActions",
                            "AveResBored",
                            "AveResEngcon",
                            "AveResConf",
                            "AveResFrust",
                            "AveResOfftask",
                            "AveResGaming"]


# In[34]:


required_cols = ['ITEST_id'] + student_specific_columns
student_specific_df = student_df[required_cols].drop_duplicates()


# In[35]:


student_specific_df.head()


# In[50]:


combined_df = pd.merge(left=training_label_df, right=student_specific_df, how='left')
X = combined_df[student_specific_columns].values
y = combined_df['isSTEM'].values


# In[60]:


combined_df = pd.merge(left=validation_test_label_df, right=student_specific_df, how='left')
X_target = combined_df[student_specific_columns].values


# # Scikit-learn method

# In[57]:


from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, mean_squared_error

sss = StratifiedShuffleSplit(n_splits=5, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    
    # test set evaluation
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    auc_test = auc(fpr, tpr)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # train set evaluation
    y_pred = model.predict(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, y_pred, pos_label=1)
    auc_train = auc(fpr, tpr)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred))
    
    print('Test: AUC: {:.5f}, RMSE: {:.5f}'.format(auc_test, rmse_test))
    print('Train:  AUC: {:.5f}, RMSE: {:.5f}'.format(auc_train, rmse_train))
    print("="*30)


# In[67]:


model = GradientBoostingClassifier()
model.fit(X, y)
y_target = model.predict_proba(X_target)


# In[75]:


# prediction submit result
result = ','.join(["{:.5f}".format(i[1]) for i in y_target])
print(result)


# # Neural Network

# In[59]:


# from keras.models import load_model
# from keras.callbacks import EarlyStopping
# from keras.models import Sequential
# from keras.layers import Dense, Reshape, Flatten
# from keras import regularizers
# from keras.optimizers import Adam
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

# def hyper_parameter_search(input_shape, num_classes):
#     num_hidden_layers = np.random.choice([3, 4], p=[0.5, 0.5])
#     reg_lambda = np.random.uniform(low=0.001, high=0.01)
#     hidden_layer_units = []
#     for i in range(num_hidden_layers):
#         # discrete uniform
#         units = np.random.randint(low=50, high=200)
#         hidden_layer_units.append(units)

#     print("num_hidden_layers:", num_hidden_layers)
#     print("lambda", reg_lambda)
#     print("hidden_layer_units", hidden_layer_units)

#     # create model
#     model = Sequential()
#     for units in hidden_layer_units:
#         model.add(Dense(units, input_dim=input_shape,
#                         kernel_regularizer=regularizers.l2(reg_lambda),
#                         activation='relu'))
#         input_shape = units

#     assert(num_classes == 1)
#     # output layer
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss="binary_crossentropy",
#                   optimizer='Adam',
#                   metrics=['accuracy'])
#     return model

