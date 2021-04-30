# Load libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from PIL import Image


np.random.seed(0)
#load the datasets for Employee who has left.
#left = pd.read_excel("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",sheet_name=2)
#left.head(4)

left = pd.read_csv("employee_who_left.csv")
left.head(4)

#load the datasets for Employee who is still existing.
#existing = pd.read_excel("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",sheet_name=1)

existing = pd.read_csv("existing_employee.csv")

## Add the atrribute Churn to Existing Employeee dataset
existing['Churn']= 'No'
existing.head(2)

## Add the attribute churn to Employee who has left dataset
left['Churn']='Yes'
left.head(2)

## Combining left and existing Dataframes together to create a single dataframes.
employee_attrition =  pd.concat([left, existing], ignore_index=True)
employee_attrition.head(10)


### Number of Employee who have left in Each department
print(left['dept'].value_counts())


# ## Data Preprocessing

# ### a) Checking For Missing Values

# Removing Redindant Variables
employee_attrition.drop('Emp ID', axis=1, inplace=True)
# Checking For Missing Values
employee_attrition.isnull().sum()
# ### b) Enconding All Categorical Variables

## Encoding The  Categorical Variables

le= LabelEncoder()

# Label Encoding will be used for columns with 2 or less unique values
le_count = 0
for col in employee_attrition.columns:
    if employee_attrition[col].dtype == 'object':
        le.fit(employee_attrition[col])
        employee_attrition[col] = le.transform(employee_attrition[col])
        le_count += 1


print('{} columns were label encoded.'.format(le_count))

## Selceting the Independent and Dependent Variables
X = employee_attrition.iloc[:,[0,1,2,3,4,5,6,7,8]].values ### Matrix of Independent faetures
y = employee_attrition.iloc[:,9].values    ### Vector of target varible 

### Splitting the data into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state= 1000)


# ##   Training and Testing the Model
from sklearn.metrics import accuracy_score,classification_report

# #### b) Random forest Classifier
from sklearn.ensemble import RandomForestClassifier

## Trainign the Model
rforest = RandomForestClassifier()
rforest.fit(X_train,y_train)

## Testing the Model
y_pred_rforest = rforest.predict(X_test)




#Only Decison tree
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

#only logistic regression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

## Testing the Model
y_pred_clf = clf.predict(X_test)


rfc_acc=accuracy_score(y_test,y_pred_rforest)
#for decsion tree
print("Decision Tree Classifier")
#print(classification_report(y_test,y_pred_clf))
clf_acc=accuracy_score(y_test,y_pred_clf)
print("Accuracy :" , clf_acc)
#for logistic regression
print("Logistic Regression")
#print(classification_report(y_test, predictions))
lr_acc=accuracy_score(y_test,predictions)
print("Accuracy :",lr_acc)

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier'],
    'Score': [lr_acc,clf_acc,rfc_acc]})
models.sort_values(by='Score', ascending=False)