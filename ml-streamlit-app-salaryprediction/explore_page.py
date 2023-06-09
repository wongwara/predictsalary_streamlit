import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/wongwara/Jobseeker_Baymax/main/dataset/listings2019_2022.csv")
    return df

df = load_data()

def show_explore_page():
    st.title("💰 Job salary for data scientist in AUSTRALIA")
  
    st.write(
        """ 
        As a data science student preparing to enter the job market, we were curious about programming languages and whether or not we required a basic understanding of multiple languages, as well as what employers expect of job candidates.
        """
    ) 
    st.write(
        """
             Therefore, the objective of this project would be to develop a machine learning model that accepts a job title and the job description with any related words from the job as input and returns expected salary associated with that job.
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
    st.write("""
    The highest demands for an Australian job classification are in the Information Communication Technology field, followed by science and technology, then Government& Defence.
    """)
    st.write(
        """
    ### Jobs released each month from January 2019 to January 2022
    """
    )
    # Compute the monthly job counts
    df['listingDate'] = pd.to_datetime(df['listingDate'])
    monthly_count = df.resample('M', on='listingDate').size().reset_index(name='count')

    # Create a Plotly line chart
    fig2 = px.line(monthly_count, x='listingDate', y='count', title='Job Release Amount per Month', width=800, height=500)

    # Update the x-axis tick labels
    fig2.update_layout(xaxis_tickangle=-45)

    # Display the chart in Streamlit
    st.plotly_chart(fig2)
    st.write(
        """
    The chart above represents the data by counting the number of jobs released each month from January 2019 to January 2022, grouping them by month, and calculating the overall number.
    """
    )
    st.write(
        """
    ### Top 10 Programming Languages Required for Data Scientist
    """
    )
    sum_programming = df.iloc[:, 24:49].sum()
    sum_programming_cleaned = pd.Series(sum_programming.loc[sum_programming >= 50])
    top10_programming_cleaned = sum_programming_cleaned.sort_values().tail(10)

    fig = px.bar(x=top10_programming_cleaned, y=top10_programming_cleaned.index, orientation='h', color=top10_programming_cleaned.index)
    fig.update_layout(title='Top 10 Programming Languages Required for Data Scientist',
                  xaxis_title='Count',
                  yaxis_title='Programming Language')
    st.plotly_chart(fig)
    st.write(
        """
        Knowing the most popular programming languages is crucial for job searchers who want to obtain significant insight into the data scientist job market. Some of the most widely used languages in the sector are Python, SQL, R, SAS, and many others. employment searchers can better understand which talents are most in demand in the current employment market by graphing the usage frequency of various languages. 
        """
    )
    # data preprocessing
    state_name = ['New South Wales', 'Australian Capital Territory', 'Victoria','Western Australia', 'Queensland', 'Northern Territory', 'South Australia', 'Tasmania']
    job_amount = df['state'].value_counts()

    df_jobinAU = pd.DataFrame(columns = ['amount'], index = state_name)

    for state in job_amount.index:
        if state in state_name:
            df_jobinAU['amount'][state] = job_amount[state]

    df_jobinAU = df_jobinAU.sort_values('amount')

    fig = px.bar(df_jobinAU, x='amount', y=df_jobinAU.index, orientation='h', color='amount', 
             color_continuous_scale='Blues', range_color=(0, max(df_jobinAU['amount'])),
             labels={'amount': 'Job Amount', 'index': 'State'})

    fig.update_layout(title='Job Amount in Different States', xaxis_title='', yaxis_title='')

    st.plotly_chart(fig)
    st.write("""the states of NSW, VIC, and QLD have the most job opportunities.""")
   
    job_vacancy = df['companyName'].value_counts()
    job_vacancy_cleaned = pd.Series(job_vacancy.loc[job_vacancy >= 10])

    fig = go.Figure(go.Bar(
        x=job_vacancy_cleaned.index,
        y=job_vacancy_cleaned,
        marker=dict(color='#275e8e'),
    ))

    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_tickfont=dict(size=10),
        margin=dict(l=20, r=20, t=50, b=20),
        title='Jobs release more than 10 vacancies in four years'
    )

    st.plotly_chart(fig)
    st.write("""
     The data shows that the Australian Bureau of Statistics, the Australian Department of Defence, SEEK, Capgemini, and Suncorp are the top five corporations that have posted the most job openings.
    """)
    
    st.write(
        """
    ### Programming Languages Demand in Different States
    """
    )
    df_lang = df.iloc[:, 24:49]
    sum_languages = df_lang.sum()
    sum_languages = sum_languages.sort_values()
    df_lang_cleaned = df_lang.loc[:, sum_languages >= 50]

    grouped = df_lang_cleaned.groupby(df['state']).sum()

    grouped = grouped.drop(['Overseas', 'UK & Ireland'], axis = 0)

    fig = go.Figure()
    for column in grouped.columns:
        fig.add_trace(go.Bar(
            x=grouped[column].values,
            y=grouped[column].index,
            name=column,
            orientation='h'
        ))

    fig.update_layout(
        barmode='stack',
        title='Programming Languages Demand in Different States',
        xaxis_title='Number of Job Openings',
        yaxis_title='State',
        )

    st.plotly_chart(fig)
    st.write("""
     The data shows the different demands of programming languages in each state.
    """)
    
    
