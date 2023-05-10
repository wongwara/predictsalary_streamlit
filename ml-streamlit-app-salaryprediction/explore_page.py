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
    chart_data = {'x': monthly_count['listingDate'], 'y': monthly_count['count']}
    st.line_chart(chart_data)
    
#     st.set_title('Job release amount per week')
#     ax.tick_params(axis='x', rotation=45)

#     st.pyplot(fig1)
