
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Libraries Import

# In[11]:


data = pd.read_csv('C:/Users/priye/Desktop/LearningSpring2018/python/Week-7-MachineLearning/weather/daily_weather.csv')


# Data Import
# 

# In[12]:


del data['number']
data = data.dropna()


# Data Cleaning

# In[13]:


data.head()


# In[16]:


data2= data.copy()
#duplicating data into another dataframe
#data2 is clean data


# In[24]:


data2['high_humidity']= (data2['relative_humidity_3pm']>24.99)*1.0
# selected data2['high_humidity'] because we want to see predict temprature of afternpoon based on morning data


# In[37]:


data2['high_humidity']
y=data2[['high_humidity']].copy()
#k is new data frame, copied the values from data2['high_humidity'] into it


# In[38]:


y.head()


# In[39]:


morning_features = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']
#filter
#using all other morning variables we want to predict values of afternoon temprarture


# In[41]:


x= data2[morning_features].copy()


# In[51]:


x.head()


# In[52]:


y.head()


# In[83]:


x_train, x_test, y_train,y_test = train_test_split(x,y,test_size= .33)

#splitting x and y dataframe into their train and test dataframes
#train data set is 67% of original dataset and test data is 33% of original dataframe


# In[90]:


classifier= DecisionTreeClassifier(max_leaf_nodes= 10)
#created a classifier using DecisionTreeClassifier function.
classifier.fit(x_train,y_train)
#tested the classifier on traing data with fit(), relating data of x_train with y_train and reviewing it


# In[91]:


type(classifier)


# In[92]:


predictions = classifier.predict(x_test)

#used predict() to predict y_test values using x_test values


# In[93]:


predictions[:10]


# In[94]:


y_test[:10]


# In[95]:


accuracy_score(y_true = y_test,y_pred= predictions)

