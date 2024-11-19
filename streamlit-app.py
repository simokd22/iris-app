# Import the required packages
import streamlit as st
import pandas as pd
import seaborn as sn
import altair as alt

# Load data
df = pd.read_csv('iris.csv', delimiter=',')

# Set page title
st.title('ISJM BI - Iris Dataset Explorer')

st.header('Iris Data pre-visualization Section')  # Sets a header for a section


# Display data
st.write(df.head())

my_dataset='iris.csv'

st.subheader('Subsection: Pie Chart Analysis')  # Sets a subheader for a subsection

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
    x='petal_length',
    y='sepal_width',
    color='species'
)

# Display chart
st.write(chart)


sn.pairplot(dataset, hue="variety")

# About

if st.button("About App"):
	st.subheader("Iris Dataset EDA App")
	st.text("Built with Streamlit")
	st.text("Thanks to the Streamlit Team Amazing Work")

if st.checkbox("By"):
	st.text("Jesse E.Agbe(JCharis)")
	st.text("Jesus Saves@JCharisTech")
