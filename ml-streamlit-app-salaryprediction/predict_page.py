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
    st.title("🤖 Salary Prediction")
    st.write(""" ##Data Scientist salary prediction using a machine learning model""")

    st.write(""" We need some information to predict the salary""")
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
    col1, col2 = st.columns(2)

    with col1:
        st.write(""" ###Job Classification and options""")
        job_classification_options = list(job_classification_dict.keys())
        job_classification = st.selectbox("jobClassification", job_classification_options)
        jobClassification = job_classification_dict[job_classification]
        state_options = list(state_dict.keys())
        state = st.selectbox("state", state_options)
        state = state_dict[state]
        isRightToWorkRequired_options = list(isRightToWorkRequired_dict.keys())
        isRightToWorkRequired = st.selectbox("isRightToWorkRequired", isRightToWorkRequired_options)
        isRightToWorkRequired = isRightToWorkRequired_dict[isRightToWorkRequired]
        recruiter_options = list(recruiter_dict.keys())
        recruiter = st.selectbox("recruiter", recruiter_options)
        recruiter = recruiter_dict[recruiter] 
        st.write(""" Australia map Photo by Jon Tyson on Unsplash
   """)
        st.image("https://images.unsplash.com/photo-1530230624258-4055a187ef65?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1341&q=80")

    with col2:
        st.write(""" ###Programming Language Required""")
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
    
#     job_classification_options = list(job_classification_dict.keys())
#     job_classification = st.selectbox("jobClassification", job_classification_options)
#     jobClassification = job_classification_dict[job_classification]
#     state_options = list(state_dict.keys())
#     state = st.selectbox("state", state_options)
#     state = state_dict[state]
#     isRightToWorkRequired_options = list(isRightToWorkRequired_dict.keys())
#     isRightToWorkRequired = st.selectbox("isRightToWorkRequired", isRightToWorkRequired_options)
#     isRightToWorkRequired = isRightToWorkRequired_dict[isRightToWorkRequired]


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
        salary_range_str = salary[0].strip('[]()')  # remove the brackets and parentheses
        salary_range_list = salary_range_str.split(',')  # split the string by comma
        min_salary = int(float(salary_range_list[0]))  # convert the first value to float and then to int
        max_salary = int(float(salary_range_list[1]))  # convert the second value to float and then to int
        st.subheader(f"The estimate salary from the given information is in the range of {min_salary:,.0f} to {max_salary:,.0f}$")
