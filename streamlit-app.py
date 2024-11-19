# Import the required packages
import streamlit as st
import pandas as pd
import altair as alt

# Load data
df = pd.read_csv('iris.csv', delimiter=',')

# Set page title
st.title('ISJM BI - Iris Dataset Explorer')

# Display data
st.write(df.head())

my_dataset='iris.csv'

# Show Dataset
if st.checkbox("Preview DataFrame"):
	data = explore_data(my_dataset)
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
    color='species'
)

# Display chart
st.write(chart)
