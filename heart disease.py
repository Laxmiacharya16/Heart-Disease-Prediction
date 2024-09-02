#!/usr/bin/env python
# coding: utf-8

# In[46]:


## importing necessary libraries


# In[ ]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


# In[47]:


dataset = pd.read_csv(r"C:/Users/srupa/Downloads/unified mentor/3rd project/Heart Disease data.csv")
dataset


# ## Data Description

# 1. Age: The age of the patient in years.
# 2. Sex: The gender of the patient (1 = male, 0 = female).
# 3. Chest Pain Type (4 values):
#     0: Typical angina
#     1: Atypical angina
#     2: Non-anginal pain
#     3: Asymptomatic
# 4. Resting Blood Pressure: The patient's resting blood pressure (in mm Hg).
# 5. Serum Cholesterol: The patient's serum cholesterol level in mg/dl.
# 6. Fasting Blood Sugar > 120 mg/dl: A binary variable indicating if the patient's fasting blood sugar is greater than 120 mg/dl (1 = true, 0 = false).
# 7. Resting Electrocardiographic Results (values 0, 1, 2):
#     0: Normal
#     1: Having ST-T wave abnormality (e.g., T wave inversions and/or ST elevation or depression of > 0.05 mV)
#     2: Showing probable or definite left ventricular hypertrophy by Estes' criteria
# 8. Maximum Heart Rate Achieved: The maximum heart rate achieved by the patient during the test.
# 9. Exercise Induced Angina: Exercise-induced angina (1 = yes, 0 = no).
# 10. Oldpeak: ST depression induced by exercise relative to rest.
# 11. Slope of the Peak Exercise ST Segment:
#     0: Upsloping
#     1: Flat
#     2: Downsloping
# 12. Number of Major Vessels (0-3) Colored by Fluoroscopy: The number of major vessels (0-3) that are colored by fluoroscopy.
# 13. Thalassemia (thal):
#     0: Normal
#     1: Fixed defect
#     2: Reversible defect
# 
# 14. Target Variable:
# 
# The target variable in this dataset typically indicates the presence or absence of heart disease. It is a binary classification variable, where:
# 
#     1 indicates the presence of heart disease.
#     0 indicates the absence of heart disease.

# In[48]:


dataset.info()


# In[49]:


## we can see there is no null values in the any column and all the column are assigned correct datatypes


# In[50]:


dataset.describe()


# In[51]:


## checking for the dublicate values
dataset.duplicated().sum()


# In[52]:


## there are 723 duplicate dataset which is very large number
# keeping duplicate values are not good so we will drop the duplicate data 


# In[53]:


dataset.drop_duplicates(inplace = True)


# In[54]:


dataset.shape


# In[55]:


## checking whether the dataset is balanced or not
dataset["target"].value_counts()


# In[56]:


## EDA


# In[57]:


plt.figure(figsize=(14, 8))
sns.countplot(x='age', hue='target', data=dataset)

# Display the plot
plt.show()


# There are notable peaks around the ages of 41, 42, 43, 51, 52, 54, 58, and 60, indicating higher counts of individuals in these age groups.
# people with age from 29 to 54 are more likely to have heart disease and above 54 people are more likely to not have the disease
# most of the people 

# In[58]:


dataset.columns


# In[15]:


sns.countplot(x = "sex", hue = "target", data = dataset)
plt.legend(title='Sex', labels=['Female', 'Male'])
plt.show()


# ##Women have a higher incidence of heart disease, while men are less likely to be affected by the condition.

# In[16]:


sns.countplot(x = "cp", hue = "target", data = dataset)
 
plt.show()


# #cp 2 (Non-anginal pain) have a higher incidence of heart disease and cp 0 (Typical angina) are less likely to be affected by the condition

# In[17]:


#A binary variable indicating if the patient's fasting blood sugar is greater than 120 mg/dl (1 = true, 0 = false).
sns.countplot(x = "fbs", hue = "target", data = dataset)
 
plt.show()


# In[18]:


# most of the people having heart disease have fasting blood sugar smaller than 120 mg/dl.


# In[19]:


plt.figure(figsize = (14,8))
sns.pairplot(dataset)


# In[59]:


# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[ ]:


## there is no high correlation between any columns 


# In[20]:


plt.figure(figsize=(8, 5))
sns.boxplot(data=dataset, x='target', y='slope')
plt.title('Slope of Peak Exercise ST Segment by Heart Disease Status')
plt.xlabel('Heart Disease Status')
plt.ylabel('Slope of Peak Exercise ST Segment')
plt.show()


# In[21]:


# box plot
num_columns = len(dataset.columns)
num_rows = (num_columns + 2) // 3  

plt.figure(figsize=(15, num_rows * 4)) 
for i, column in enumerate(dataset.columns):
    plt.subplot(num_rows, 3, i + 1)  
    sns.boxplot(y=dataset[column])
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)

plt.tight_layout()
plt.show()


# There is outliers present in some columns

# In[83]:


## splitting x and y variables
x = dataset.drop("target", axis = 1)
y = dataset["target"]


# In[85]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)


# In[86]:


x_train.shape, x_test.shape


# In[87]:


## scaling the data 


# In[88]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)


# In[89]:


x_test_scaled = scaler.transform(x_test)


# In[26]:


##training random forest, decision tree, svm, lr, xgboost and adaboost 


# In[66]:


models = {'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'XGBoost': XGBClassifier(),
    'AdaBoost': AdaBoostClassifier()
    }


# In[67]:


for name, model in models.items():
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.2f}')


# In[30]:


## we are getting highest accuracy from the random forest so lets try some fine tuning for increasing the accuracy 


# In[68]:


param_grid = {
    'n_estimators': [50, 100, 200],
    'criterion': ["gini", "entropy","log_loss"],
    'max_features': ['auto', 'sqrt', 'log2'], 
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'bootstrap': [True, False]  
}


# In[94]:


rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, scoring='accuracy', 
                           n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(x_train_scaled, y_train)

# Print best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[95]:


# Predict on the test set
best_random = grid_search.best_estimator_
y_pred = best_random.predict(x_test_scaled)
# Print accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(accuracy))


# In[72]:


best_random.feature_importances_.tolist()


# In[73]:


feature_important = pd.DataFrame({
    'column':x.columns,
    'value':best_random.feature_importances_.tolist()
})

feature_important


# In[74]:


plt.figure(figsize=(25,8))
plt.bar(feature_important.column.values.tolist(), feature_important.value)
plt.title("Feature Importances")
plt.show()


# In[ ]:


## we can see cp, oldpeak and thal is considered most important columns for predicting the output


# In[101]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)



# In[102]:


# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)


# In[96]:


# Save the best_random model
with open('bestrf_model.pkl', 'wb') as file:
    pickle.dump(best_random, file)


# In[99]:


cwd = os.getcwd()
cwd


# In[103]:


with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# In[ ]:





# In[ ]:




