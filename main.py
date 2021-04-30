# Load libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from PIL import Image

st.markdown(
    """
    <style>
    .main{
        background-color : #F5F5F5
    }
    </style>
    """,
    unsafe_allow_html=True
)

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


#### Tile For the Web App
st.title(
         """
            Employee Turnover Predictor Web App
         """)
st.write(
         """
         Employee attrition occurs when the size of your workforce diminishes over time due to unavoidable factors 
         such as employee resignation for personal or professional reasons.
         Employees are leaving the workforce faster than they are hired, and it is often outside the employer’s control.
         """)

st.write(
         """
         This Program is developed using the Machine Learning Algorithm using Random Forest Classifier,
         it predicts the Employee who will leave the organization.
         """)

## Display
image = Image.open("pic.jpg")

st.image(image,use_column_width=True)

### Subtitile
st.subheader('Dataset:')

st.dataframe(employee_attrition.sample(frac=1).head(25))

##
st.header(
    """
    Exploratory Data Analysis of Employee Who Left
    The Descriptive Summary of the Employee who have left based on Departmemt
    """
)

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
#clf = DecisionTreeClassifier()
#clf.fit(X_train,y_train)
#y_pred_clf = clf.predict(X_test)

#only logistic regression
#regressor = LogisticRegression()
#regressor.fit(X_train, y_train)
#predictions = regressor.predict(X_test)


# ### Predicting the Existing Employee with the Probability of leaving
st.sidebar.header("Prediction Section")
st.sidebar.subheader(
"""
Enter Value For the Features Below
""")
def user_input():
    Satisfaction_level = st.sidebar.number_input("Satisfaction level",min_value=0.00, max_value= 0.99,value=0.5)
    
    Last_evaluation = st.sidebar.number_input("Last Evaluation",min_value=0.00, max_value= 0.99,value=0.5)
    
    number_project =st.sidebar.number_input('Number of project',min_value=0, max_value= 10,value=5)
    
    average_montly_hours = st.sidebar.number_input('The average montly hours',min_value=0.00,max_value= 1000.00,value=300.00)
    
    time_spend_company  = st.sidebar.number_input('Time spend in company',min_value=0, max_value= 20,value=5)
    
    Work_accident =st.sidebar.selectbox('Work accident',(0, 1))
    
    promotion_last_5years = st.sidebar.selectbox('Promotion last 5 years',(0, 1))
    
    dept = st.sidebar.selectbox('Department',("sales","technical","support","IT","hr","accounting",
                                              "marketing","product_mng","randD","mangement"))
    
    Salary =  st.sidebar.selectbox('Salary Level ',("low","medium","high"))
    
    ### Dictionaries of Input
    input_user= {"Satisfaction_level":Satisfaction_level ,
                 "Last_evaluation":Last_evaluation,
                 "number_project":number_project,
                 "average_montly_hours":average_montly_hours,
                 "time_spend_company":time_spend_company,
                 "Work_accident":Work_accident,
                 "promotion_last_5years":promotion_last_5years,
                 "dept":dept,"Salary":Salary}
               
    ### Cpnverting to a Dataframes
    input_user =pd.DataFrame(input_user,index=[0])
    return input_user

input_value = user_input()      

# Label Encoding will be used for columns with 2 or less unique values
## Encoding The  Categorical Variables

le1= LabelEncoder()

le1_count = 0
for col in input_value.columns:
    if input_value[col].dtypes == 'object':
        le1.fit(input_value[col])
        input_value[col] = le1.transform(input_value[col])
        le1_count += 1


#print('{} columns were label encoded.'.format(le1_count))
## Chart
fig1,ax= plt.subplots(figsize=(12,5))
ax.bar(left['dept'].value_counts().index,left['dept'].value_counts().values)
plt.ylabel("Number of Employee who have left")
plt.xlabel("Departments")
plt.title("What departments are they from?")
plt.grid()
st.pyplot(fig1)

st.write("The Descriptive Summary of the Employee who have left based on Monthly Work Hours")
fig2,ax= plt.subplots(figsize=(12,5))
ax.bar(left['average_montly_hours'].value_counts().index,left['average_montly_hours'].value_counts().values)
plt.ylabel("Number of Employee who have left")
plt.xlabel("Number of Working Hours")
plt.title("Number of Monthly hours put in by the employee")
plt.grid()
st.pyplot(fig2)

st.write("The Descriptive Summary of the Employee who have left based on Time In Company")
fig3,ax= plt.subplots(figsize=(12,5))
ax.bar(left['time_spend_company'].value_counts().index,left['time_spend_company'].value_counts().values)
plt.ylabel("Number of Employee who have left")
plt.xlabel("Years at Company")
plt.title("Years spent in the company")
plt.grid()
st.pyplot(fig3)

st.write("The Descriptive Summary of the Employee who have left based on No.of Projects Worked")
fig4,ax= plt.subplots(figsize=(12,5))
ax.bar(left['number_project'].value_counts().index,left['number_project'].value_counts().values)
plt.ylabel("Number of Employee who have left")
plt.xlabel("Number of Projects")
plt.title("Number of projects worked on")
plt.grid()
st.pyplot(fig4)


st.write("The Descriptive Summary of the Employee who have left based on Salary")
fig5,ax= plt.subplots(figsize=(12,5))
ax.bar(left['salary'].value_counts().index,left['salary'].value_counts().values)
plt.ylabel("Number of Employee who have left")
plt.xlabel("Salary")
plt.title("What was the range of salary?")
plt.grid()
st.pyplot(fig5)

#Feature Importance
employee_attrition =employee_attrition[['Churn','satisfaction_level','last_evaluation',
                                        'number_project','average_montly_hours',
                                        'time_spend_company','Work_accident',
                                        'promotion_last_5years','dept','salary']]
importances = pd.DataFrame({'feature':employee_attrition.iloc[:, 1:employee_attrition.shape[1]].columns,
                            'importance':np.round(rforest.feature_importances_,3)}) 
#Note: The target column is at position 0
importances = importances.sort_values('importance',ascending=False).set_index('feature')
st.subheader('Feature Importance based on Analysis')
st.write(importances)
st.bar_chart(importances)


st.header(
"""
Evaluation Metrics
""")
## Classification Report
st.subheader("Logistic Regression : 78 % ""accuracy")

st.subheader("Decision Tree Classifier : 97 % ""accuracy")

st.subheader("Random Forest Classifier: 99 % ""accuracy")
print("Random Forest Classifier")
st.write(classification_report(y_test,y_pred_rforest))
print(classification_report(y_test,y_pred_rforest))
print(accuracy_score(y_test,y_pred_rforest))
#Conclusions
st.header("Conclusions")
st.write("""
         Monthly Income: people on higher wages are less likely to leave the company. 
         Hence, efforts should be made to gather information on industry benchmarks in the current 
         local market to determine if the company is providing competitive wages.""")
st.write("""
         Over Time: people who work overtime are more likely to leave the company. 
         Hence efforts must be taken to appropriately scope projects upfront with adequate support 
         and manpower so as to reduce the use of overtime.""")
st.write("""YearsAtCompany: Senior Employees are less likely to leave. Employees who hit their 
         two-year anniversary should be identified as potentially having a higher-risk of leaving.""")
st.write("""Patterns in the employees who have resigned: this may indicate recurring patterns in employees
         leaving in which case action may be taken accordingly.""")

#models = pd.DataFrame({
 #   'Model': ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier'],
  #  'Score': [lr_acc,clf_acc,rfc_acc]})
#models.sort_values(by='Score', ascending=False)

#st.dataframe(models)



if st.sidebar.button("Predict"):
    Prediction = rforest.predict(input_value)
    if Prediction == 0:
        result = pd.DataFrame({"Churn":Prediction,"Info":"The Employee will not Leave the Ogarnization"})
    else:
        result = pd.DataFrame({"Churn":Prediction,"Info":"The Employee wil Leave the Ogarnization"})
    
    st.title('Predictions')
    st.subheader('Custom Input Values')
    st.dataframe(input_value)                          
    
    st.write("""
            # The Result of the Classification:
             """)
    st.write("Attrition : ") 
    st.dataframe(result)
                             


                       
                       
