import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/wongwara/Jobseeker_Baymax/main/dataset/final_cleaned.csv', index_col=[0])

#code for encoder only the columns that need to transform
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
jobClassification_cats = ['Information & Communication Technology',
       'Banking & Financial Services', 'Science & Technology',
       'Education & Training', 'Government & Defence',
       'Consulting & Strategy', 'Healthcare & Medical',
       'Human Resources & Recruitment', 'Marketing & Communications',
       'Retail & Consumer Products', 'Administration & Office Support',
       'Accounting', 'Insurance & Superannuation',
       'Mining, Resources & Energy', 'Real Estate & Property',
       'Manufacturing, Transport & Logistics', 'Engineering']

state_cats = [['Australian Capital Territory', 'South Australia','Western Australia']]
df = df.drop(['teaser','desktopAdTemplate','state_encoded','min_salary','max_salary','workType','salary_section_enc'],axis =1)

# Display the resulting dataframe
print(df.head())

# Separate to X and y
y = df.pop('salary_section')
X = df
from sklearn.model_selection import train_test_split
# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.columns = X_train.columns.astype(str) 
X_test.columns = X_test.columns.astype(str)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score
# Model building

# create an SVM model with the best hyperparameters found using grid search
svm_model = SVC(C=10, gamma='scale', kernel='linear')

# fit the SVM model to the training data
svm_model.fit(X_train, y_train)

# make predictions on the test data using the trained SVM model
y_pred = svm_model.predict(X_test)

# calculate the accuracy of the SVM model predictions on the test data
accuracy = accuracy_score(y_test, y_pred)

jobClassification_enc = OrdinalEncoder()


import pickle
data = {"model": svm_model, "encode": jobClassification_enc}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

regressor_loaded = data["model"]
jobClassification_enc = data["encode"]

# y_pred = regressor_loaded.predict(X)
# y_pred
