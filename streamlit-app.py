# Import the required packages
import streamlit as st
import pandas as pd
import altair as alt

# Load data
df = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv', delimiter=',')

# Set page title
st.title('Iris Dataset Explorer')

# Display data
st.write(df)

# Create chart
chart = alt.Chart(df).mark_point().encode(
    x='petal.length',
    y='sepal.width',
    color='variety'
)

# Display chart
st.write(chart)
