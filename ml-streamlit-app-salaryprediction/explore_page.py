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

    st.write(
        """
    ### Jobs released each month from January 2019 to January 2022
    """
    )
    df.index = pd.to_datetime(df['listingDate'])
    monthly_count = df.resample('W').size()
    monthly_count = monthly_count.reset_index(name = 'count')

    plt.bar(monthly_count['listingDate'], monthly_count['count'], width = 6, color = ["#275e8e"])
    plt.title('Job release amount per week')
    plt.xticks(rotation = 45)
    plt.show()
