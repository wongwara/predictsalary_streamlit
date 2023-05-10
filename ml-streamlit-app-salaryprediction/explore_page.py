import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

@st.cache
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/wongwara/Jobseeker_Baymax/main/dataset/listings2019_2022.csv")
    return df

df = load_data()

def show_explore_page():
    st.title("💰 Job salary for data scientist in AUSTRALIA")
    
    
    st.write(
        """
    ### Jobs released each month from January 2019 to January 2022
    """
    )
    st.subheader("The highest demand job classification in Australia")
    # Compute the counts for each job classification
    job_counts = df['jobClassification'].value_counts()
    # Select the top 10 highest job counts
    top_job_counts = job_counts.nlargest(10)


    # Create a Plotly bar chart
    fig = px.bar(x=top_job_counts.index, y=top_job_counts.values, color=top_job_counts.index)
    fig.update_layout(xaxis_title='Job Classification', yaxis_title='Count', title='Top 10 Highest Demand Job Classification')

    # Display the chart in Streamlit
    st.plotly_chart(fig)
