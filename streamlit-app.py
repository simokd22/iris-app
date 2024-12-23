# Import the required packages
import streamlit as st
import pandas as pd
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Iris Classification", 
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# -------------------------
# Sidebar

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:

    st.title('Iris Classification')

    # Page Button Navigation
    st.subheader("Pages")
    st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',))
    st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',))
    st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',))
    st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',))
    st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',))
    st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',))
    st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',))

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of training classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
    st.markdown("by: [Zeraphim](https://jcdiamante.com)")

# -------------------------

# Load data
try:
    df = pd.read_csv('iris.csv', delimiter=',')
except FileNotFoundError:
    st.error("Dataset not found. Ensure 'iris.csv' is available in the app directory.")
    st.stop()

# Render pages based on session state
if st.session_state.page_selection == 'about':
    st.title('About the Iris Classification App')
    st.markdown("This app demonstrates the use of classification models on the Iris dataset.")
    st.markdown("Explore different pages using the sidebar.")

elif st.session_state.page_selection == 'dataset':
    st.title('Dataset')
    st.write(df)

elif st.session_state.page_selection == 'eda':
    st.title('Exploratory Data Analysis')
    st.subheader('Data Description')
    st.write(df.describe())
    st.subheader('Correlation Matrix')
    st.write(df.corr())
    st.subheader('Charts')
    chart = alt.Chart(df).mark_point().encode(
        x='petal_length',
        y='petal_width',
        color="species"
    )
    st.altair_chart(chart, use_container_width=True)

elif st.session_state.page_selection == 'data_cleaning':
    st.title('Data Cleaning / Pre-processing')
    st.write("This section will show the steps for cleaning and preprocessing the Iris dataset.")
    # Add your data cleaning logic here.

elif st.session_state.page_selection == 'machine_learning':
    st.title('Machine Learning')
    st.write("This section will show machine learning model training and evaluation.")
    # Add your machine learning model code here.

elif st.session_state.page_selection == 'prediction':
    st.title('Prediction')
    st.write("This section allows you to make predictions using the trained models.")
    # Add your prediction logic here.

elif st.session_state.page_selection == 'conclusion':
    st.title('Conclusion')
    st.markdown("The Iris classification app demonstrates how to build an interactive dashboard with Streamlit.")

# Footer Section
st.sidebar.subheader("Contact")
st.sidebar.text("Stéphane C. K. Tékouabou")
st.sidebar.text("ctekouaboukoumetio@gmail.com")

if st.checkbox("By"):
	st.text("Stéphane C. K. Tékouabou")
	st.text("ctekouaboukoumetio@gmail.com")
