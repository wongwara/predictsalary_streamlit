import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

@st.cache
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/wongwara/Jobseeker_Baymax/main/dataset/listings2019_2022.csv")
    return df

df = load_data()

def show_explore_page():
    st.title("Explore Software Engineer Salaries")
    
    st.dataframe(df)
    st.write(
        """
    ### Jobs released each month from January 2019 to January 2022
    """
    )
    
