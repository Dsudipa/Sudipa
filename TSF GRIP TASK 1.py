#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION 

# TASK 1

# # MADE BY - SUDIPA DUTTA

# # OBJECTIVE- Predicting student's score based on their study time 

# The task is to predict the score of students who studied 9.25 hrs per day, here we have to use linear reggression as our data consists of two variables

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[13]:


# Create a DataFrame from the provided data
data = pd.DataFrame({
    'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.8],
    'Scores': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 30, 54, 35, 76, 86]
})




# In[14]:


# Check the first few rows of the dataset
print(data.head())


# In[15]:


# Visualize the relationship between study hours and percentage
plt.scatter(data['Hours'], data['Scores'])
plt.title('Study Hours vs. Percentage')
plt.xlabel('Study Hours')
plt.ylabel('Percentage')
plt.show()


# In[16]:


# Split the data into training and testing sets
X = data[['Hours']]
y = data['Scores']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[18]:


# Make predictions on the testing data
y_pred = model.predict(X_test)


# In[19]:


# Plot the regression line along with the data points
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Study Hours vs. Percentage (Regression Line)')
plt.xlabel('Study Hours')
plt.ylabel('Percentage')
plt.show()


# In[26]:


# Check the first few rows of the dataset
print(data.head())


# In[27]:


#comparing actual vs predicted 
df= pd.DataFrame({'Actual':y_test, 'predicted':y_pred})


# In[28]:


df


# Evaluting the model 

# In[32]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))

