from sklearn.preprocessing import OrdinalEncoder
import streamlit as st
import pandas as pd
from prediction import load_model
from sklearn.svm import SVC
import re

data = load_model()
regressor_loaded = data["model"]
jobClassification_enc = data["encode"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")
    job_classification_dict = {
    'Information & Communication Technology': 0,
    'Banking & Financial Services': 1,
    'Science & Technology': 2,
    'Education & Training': 3,
    'Government & Defence': 4,
    'Consulting & Strategy': 5,
    'Healthcare & Medical': 6,
    'Human Resources & Recruitment': 7,
    'Marketing & Communications': 8,
    'Retail & Consumer Products': 9,
    'Administration & Office Support': 10,
    'Accounting': 11,
    'Insurance & Superannuation': 12,
    'Mining, Resources & Energy': 13,
    'Real Estate & Property': 14,
    'Manufacturing, Transport & Logistics': 15,
    'Engineering': 16}
    state_dict = {'Australian Capital Territory': 0,
                  'South Australia': 1,
                  'Western Australia': 2}
    
    isRightToWorkRequired_dict = {
        'No': 0,
        'Yes': 1,
    }

    Python_dict = {
        'No': 0,
        'Yes': 1,
    }
    
    SQL_dict = {
        'No': 0,
        'Yes': 1,
    }
    
    R_dict = {
        'No': 0,
        'Yes': 1,
    }
    
    Tableau_dict = {
        'No': 0,
        'Yes': 1,
    }
    
    SAS_dict = {
        'No': 0,
        'Yes': 1,
    }
    
    Matlab_dict = {
        'No': 0,
        'Yes': 1,
    }
    
    Hadoop_dict = {
        'No': 0,
        'Yes': 1,
    }
    
    Spark_dict = {
        'No': 0,
        'Yes': 1,
    }
    
    Java_dict = {
        'No': 0,
        'Yes': 1,
    }
    
    Scala_dict = {
        'No': 0,
        'Yes': 1,
    }
    
    recruiter_dict = {
        'No': 0,
        'Yes': 1,
    }
    
    job_classification_options = list(job_classification_dict.keys())
    job_classification = st.selectbox("jobClassification", job_classification_options)
    jobClassification = job_classification_dict[job_classification]
    state_options = list(state_dict.keys())
    state = st.selectbox("state", state_options)
    state = state_dict[state]
    isRightToWorkRequired_options = list(isRightToWorkRequired_dict.keys())
    isRightToWorkRequired = st.selectbox("isRightToWorkRequired", isRightToWorkRequired_options)
    isRightToWorkRequired = isRightToWorkRequired_dict[isRightToWorkRequired]
    Python_options = list(Python_dict.keys())
    Python = st.selectbox("Python", Python_options)
    Python = Python_dict[Python]
   
    SQL_options = list(SQL_dict.keys())
    SQL = st.selectbox("SQL", SQL_options)
    SQL = SQL_dict[SQL]
    
    R_options = list(R_dict.keys())
    R = st.selectbox("R", R_options)
    R = R_dict[R]
    
    Tableau_options = list(Tableau_dict.keys())
    Tableau = st.selectbox("Tableau", Tableau_options)
    Tableau = Tableau_dict[Tableau]
    
    SAS_options = list(SAS_dict.keys())
    SAS = st.selectbox("SAS", SAS_options)
    SAS = SAS_dict[SAS]
    
    Matlab_options = list(Matlab_dict.keys())
    Matlab = st.selectbox("Matlab", Matlab_options)
    Matlab = Matlab_dict[Matlab] 
    
    Hadoop_options = list(Hadoop_dict.keys())
    Hadoop = st.selectbox("Hadoop", Hadoop_options)
    Hadoop = Hadoop_dict[Hadoop] 
    
    Spark_options = list(Spark_dict.keys())
    Spark = st.selectbox("Spark", Spark_options)
    Spark = Spark_dict[Spark] 
    
    Java_options = list(Java_dict.keys())
    Java = st.selectbox("Java", Java_options)
    Java = Java_dict[Java] 
    
    Scala_options = list(Scala_dict.keys())
    Scala = st.selectbox("Scala", Scala_options)
    Scala = Scala_dict[Scala] 
    
    recruiter_options = list(recruiter_dict.keys())
    recruiter = st.selectbox("recruiter", recruiter_options)
    recruiter = recruiter_dict[recruiter] 

    ok = st.button("Calculate Salary")
    if ok:
        X = pd.DataFrame({
        'jobClassification': [jobClassification],
        'state': [state],
        'isRightToWorkRequired': [isRightToWorkRequired],
        'Python': [Python],
        'SQL': [SQL],
        'R': [R],
        'Tableau': [Tableau],
        'SAS': [SAS],
        'Matlab': [Matlab],
        'Hadoop': [Hadoop],
        'Spark': [Spark],
        'Java': [Java],
        'Scala': [Scala],
        'recruiter': [recruiter],
        })
        
        salary = regressor_loaded.predict(X)
        st.subheader(f"The estimated salary range is {salary}")
        salary_range_str = salary[0].strip('[]()')  # remove the brackets and parentheses
        salary_range_list = salary_range_str.split(',')  # split the string by comma
        min_salary = int(float(salary_range_list[0]))  # convert the first value to float and then to int
        max_salary = int(float(salary_range_list[1]))  # convert the second value to float and then to int
        st.subheader(f"{min_salary} to {max_salary}$")
        st.write("'(100000.0, 110000.0] :0 ', '(90000.0, 100000.0] :1', '(110000.0, 120000.0] :2 ', '(80000.0, 90000.0] :3', '(130000.0, 140000.0] :4', '(60000.0, 80000.0] :5', '(120000.0, 130000.0] :6', '(140000.0, 160000.0] :7', '(180000.0, inf] :8', '(160000.0, 180000.0] :9', '(18000.0, 60000.0] :10' ")
      

