import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
import nltk

page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()

# Page configuration
st.set_page_config(
     page_title='Salary Prediction App',
     page_icon='🌷',
     layout='wide',
     initial_sidebar_state='expanded')

# Title of the app
st.title('Salary Prediction App')
show_predict_page()

    
