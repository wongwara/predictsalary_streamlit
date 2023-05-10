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
    'Engineering': 16
}
#   jobClassification = (
#        '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17'
#     )
    
    state = (
        '0',
        '1',
        '2',
    ) 
    
    isRightToWorkRequired = (
        '0',
        '1',
    )

    Python = (
        '0',
        '1',
    )
    
    SQL = (
        '0',
        '1',
    )
    
    R = (
        '0',
        '1',
    )
    
    Tableau = (
        '0',
        '1',
    )
    
    SAS = (
        '0',
        '1',
    )
    
    Matlab = (
        '0',
        '1',
    )
    
    Hadoop = (
        '0',
        '1',
    )
    
    Spark = (
        '0',
        '1',
    )
    
    Java = (
        '0',
        '1',
    )  
    
    Scala = (
        '0',
        '1',
    )
    
    recruiter = (
        '0',
        '1',
    )
    job_classification_options = list(job_classification_dict.keys())
    job_classification = st.selectbox("jobClassification", job_classification_options)
    jobClassification = job_classification_dict[job_classification]

#     jobClassification = st.selectbox("jobClassification", jobClassification)
    
    isRightToWorkRequired = st.selectbox("isRightToWorkRequired", isRightToWorkRequired)
    st.write("f': 0, 't': 1")
    
    state = st.selectbox("state", state)
    st.write("'Australian Capital Territory':0, 'South Australia':1,'Western Australia':2")
    
    Python = st.selectbox("Python", Python)
    st.write("'Yes':1, 'No':0")
    
    SQL = st.selectbox("SQL", SQL)
    st.write("'Yes':1, 'No':0")
    
    R = st.selectbox("R", R)
    st.write("'Yes':1, 'No':0")
    
    Tableau = st.selectbox("Tableau", Tableau)
    st.write("'Yes':1, 'No':0")
    
    SAS = st.selectbox("SAS", SAS)
    st.write("'Yes':1, 'No':0")
    
    Matlab = st.selectbox("Matlab", Matlab)
    st.write("'Yes':1, 'No':0")
    
    Hadoop = st.selectbox("Hadoop", Hadoop)
    st.write("'Yes':1, 'No':0")
    
    Spark = st.selectbox("Spark", Spark)
    st.write("'Yes':1, 'No':0")
    
    Java = st.selectbox("Java", Java)
    st.write("'Yes':1, 'No':0")
    
    Scala = st.selectbox("Scala", Scala)
    st.write("'Yes':1, 'No':0")
    
    recruiter = st.selectbox("recruiter", recruiter)
    st.write("'Yes':1, 'No':0")
    
#     # create an SVM model with the best hyperparameters found using grid search
#     svm_model = SVC(C=10, gamma='scale', kernel='linear')
#     svm_model.fit(X, y)

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

#         # Output prediction
#         X['jobClassification'] = 'jobClassification'
#         X['jobClassification'] = X['jobClassification'].replace(jobClassification)


        salary = regressor_loaded.predict(X)
        st.subheader(f"The estimated salary range is {salary}")
        st.write("'(100000.0, 110000.0] :0 ', '(90000.0, 100000.0] :1', '(110000.0, 120000.0] :2 ', '(80000.0, 90000.0] :3', '(130000.0, 140000.0] :4', '(60000.0, 80000.0] :5', '(120000.0, 130000.0] :6', '(140000.0, 160000.0] :7', '(180000.0, inf] :8', '(160000.0, 180000.0] :9', '(18000.0, 60000.0] :10' ")

#     if ok:
#         X = pd.DataFrame({
#         'jobClassification': [jobClassification],
#         'IsRightToWorkRequired': [isRightToWorkRequired],
#         'State': [State],
#         'Python': [Python],
#         'SQL': [SQL],
#         'R': [R],
#         'Tableau': [Tableau],
#         'SAS': [SAS],
#         'Matlab': [Matlab],
#         'Hadoop': [Hadoop],
#         'Spark': [Spark],
#         'Java': [Java],
#         'Scala': [Scala],
#         'Recruiter': [recruiter],
#         'Teaser': [teaser],
#         'DesktopAdTemplate': [desktopAdTemplate]
#         })

#         X['jobClassification'] = OrdinalEncoder().fit_transform(X['jobClassification'])
        
#         X['IsRightToWorkRequired'] = jobClassification_enc.fit_transform(X['IsRightToWorkRequired'])
        
#         X['State'] = jobClassification_enc.fit_transform(X['State'])
        
#         X['Recruiter'] = jobClassification_enc.fit_transform(X['Recruiter'])
        
#         X['Teaser'] = preprocess_text_input(X['Teaser'])
        
#         X['DesktopAdTemplate'] = preprocess_text_input(X['DesktopAdTemplate'])
        
#         salary = svm_model.predict(X)
#         st.subheader(f"The estimated salary range is ${salary[0]:.2f}")
#         st.write("'(100000.0, 110000.0] :0 ', '(90000.0, 100000.0] :1', '(110000.0, 120000.0] :2 ', '(80000.0, 90000.0] :3', '(130000.0, 140000.0] :4', '(60000.0, 80000.0] :5', '(120000.0, 130000.0] :6', '(140000.0, 160000.0] :7', '(180000.0, inf] :8', '(160000.0, 180000.0] :9', '(18000.0, 60000.0] :10' ")
        

