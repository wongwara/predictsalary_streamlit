import streamlit as st
from predict_page import show_predict_page
import nltk

# Page configuration
st.set_page_config(
     page_title='Salary Prediction App',
     page_icon='🌷',
     layout='wide',
     initial_sidebar_state='expanded')

# Title of the app
st.title('🌷 Salary Prediction App')
show_predict_page()

    
