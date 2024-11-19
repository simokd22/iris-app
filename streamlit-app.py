# Import the required packages
import streamlit as st
import pandas as pd
import altair as alt
#import seaborn as sn

# Load data
df = pd.read_csv('iris.csv', delimiter=',')

# Set page title
st.title('ISJM BI - Exploration des données des Iris')

st.header('Pré-analyse visuelles données données des Iris TP1')  # On définit l'en-tête d'une section


# Afficher les premières lignes des données chargées data
#st.write(df.head())
	
st.subheader('Description des données')  # Sets a subheader for a subsection

# Show Dataset
if st.checkbox("Boutons de prévisualisation du DataFrame"):
	if st.button("Head"):
		st.write(df.head(2))
	if st.button("Tail"):
		st.write(df.tail())
	if st.button("Infos"):
		st.write(df.info())
	if st.button("Shape"):
		st.write(df.shape)
	else:
		st.write(df.head(2))


# Create chart
chart = alt.Chart(df).mark_point().encode(
    x='petal_length',
    y='petal_width',
    color=["#ff2b2b", "#faca2b", "#09ab3b"],
)

# Display chart
st.write(chart)


# About

if st.button("About App"):
	st.subheader("App d'exploration des données des Iris")
	st.text("Contruite avec Streamlit")
	st.text("Thanks to the Streamlit Team Amazing Work")

if st.checkbox("By"):
	st.text("Stéphane C. K. Tékouabou")
	st.text("ctekouaboukoumetio@gmail.com")
