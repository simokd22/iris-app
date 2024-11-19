# Import the required packages
import streamlit as st
import pandas as pd
import altair as alt

# Load data
df = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv', delimiter=',')

# Set page title
st.title('ISJM BI - Iris Dataset Explorer')

# Display data
st.write(df.head())

# Show Dataset
if st.checkbox("Preview DataFrame"):
	data = explore_data(df)
	if st.button("Head"):
		st.write(data.head())
	if st.button("Tail"):
		st.write(data.tail())
	else:
		st.write(data.head(2))

# Create chart
chart = alt.Chart(df).mark_point().encode(
    x='petal.length',
    y='sepal.width',
    color='variety'
)

# Display chart
st.write(chart)
