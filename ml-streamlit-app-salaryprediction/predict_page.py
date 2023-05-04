# Import OrdinalEncoder from sklearn.preprocessing
from sklearn.preprocessing import OrdinalEncoder
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from prediction import load_model

data = load_model()
regressor_loaded = data["model"]
jobClassification_enc = data["encode"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    jobClassification = (
        'Information & Communication Technology',
       'Banking & Financial Services', 'Science & Technology',
       'Education & Training', 'Government & Defence',
       'Consulting & Strategy', 'Healthcare & Medical',
       'Human Resources & Recruitment', 'Marketing & Communications',
       'Retail & Consumer Products', 'Administration & Office Support',
       'Accounting', 'Insurance & Superannuation',
       'Mining, Resources & Energy', 'Real Estate & Property',
       'Manufacturing, Transport & Logistics', 'Engineering',
    )
    
    isRightToWorkRequired = (
        '0',
        '1',
    )

    State = (
        '0',
        '1',
        '2',
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
    
    jobClassification = st.selectbox("jobClassification", jobClassification)
    isRightToWorkRequired = st.selectbox("isRightToWorkRequired", isRightToWorkRequired)
    st.write("f': 0, 't': 1")
    State = st.selectbox("State", State)
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
    
    # Input widgets
    teaser = st.text_input(
        "Enter teaser from your search , if None please type (-)",'')
    desktopAdTemplate = st.text_input(
        "Enter desktopAdTemplate from your search , if None please type (-)",'')

    def preprocess_text_input(input_str):
        # Clean the text data
        input_str = input_str.replace('[^\w\s]', '') # Remove punctuation
        input_str = input_str.replace('\d+', '') # Remove digits
        # Normalize the text data
        stop_words = set(stopwords.words('english'))
        input_str = ' '.join([word.lower() for word in input_str.split() if word.lower() not in stop_words])
        # Tokenize the text data
        input_str = word_tokenize(input_str)
        # Apply stemming
        stemmer = PorterStemmer()
        input_str = [stemmer.stem(word) for word in input_str]

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        input_tfidf = vectorizer.fit_transform([' '.join(input_str)])

        return input_tfidf.toarray()
    
    ok = st.button("Calculate Salary")
    if ok:
        X = pd.DataFrame({
        'jobClassification': [jobClassification],
        'IsRightToWorkRequired': [isRightToWorkRequired],
        'State': [State],
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
        'Recruiter': [recruiter],
        'Teaser': [teaser],
        'DesktopAdTemplate': [desktopAdTemplate]
        })

        X['jobClassification'] = jobClassification_enc.transform(X['jobClassification'])
        X['Teaser'] = preprocess_text_input(X['Teaser'])
        X['DesktopAdTemplate'] = preprocess_text_input(X['DesktopAdTemplate'])

        X = X.astype(str)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary range is ${salary[0]:.2f}")
        st.write("'(100000.0, 110000.0] :0 ', '(90000.0, 100000.0] :1', '(110000.0, 120000.0] :2 ', '(80000.0, 90000.0] :3', '(130000.0, 140000.0] :4', '(60000.0, 80000.0] :5', '(120000.0, 130000.0] :6', '(140000.0, 160000.0] :7', '(180000.0, inf] :8', '(160000.0, 180000.0] :9', '(18000.0, 60000.0] :10' ")
        

