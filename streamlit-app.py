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
    st.title('Dataset Exploration')
    # Buttons for dataset exploration options 
    option = st.radio(
        "Choose an action for dataset exploration:",
        ("View Full Dataset", "View Dataset Statistics", "View Dataset Information")
    )

    if option == "View Full Dataset":
        st.subheader("Full Dataset")
        st.dataframe(df, use_container_width=True)

        # Adding row count
        st.write(f"Number of rows: {len(df)}")

        # Adding download option
        @st.cache_data
        def convert_df_to_csv(dataframe):
            return dataframe.to_csv(index=False).encode('utf-8')

        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="Download Dataset as CSV",
            data=csv_data,
            file_name="iris_dataset.csv", 
            mime="text/csv"
        )

        st.subheader("Types des colonnes ")
        st.write(df.dtypes)
        
        st.subheader("Dimensions du dataset")
        st.write(df.shape)



    elif option == "View Dataset Statistics":
        st.subheader("Dataset Statistics")
        st.write("Numerical Columns Statistics")
        st.write(df.describe())
   

        st.write("Non-Numerical Columns Overview")
        for col in df.select_dtypes(include=['object']).columns:
            st.write(f"{col}: {df[col].nunique()} unique values")
            st.write(df[col].value_counts())


    elif option == "View Dataset Information":
        st.subheader("Dataset Information")

        # Dataset shape
        st.write(f"*Shape of Dataset:* {df.shape}")

        # Null values count
        st.write("*Missing Values in Dataset:*")
        st.write(df.isnull().sum())


elif st.session_state.page_selection == 'eda':
    st.title('Exploratory Data Analysis')
    
    # Data Description
    st.subheader('Data Description')
    st.write(df.describe())
    
    # Correlation Matrix
    st.subheader('Correlation Matrix (Numeric Columns)')
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        st.write(numeric_df.corr())
    else:
        st.write("No numeric columns available for correlation matrix.")
    
    # Charts
    st.subheader('Charts')
    st.markdown("### Petal Length vs Petal Width (by Species)")
    chart = alt.Chart(df).mark_point().encode(
        x='petal_length',
        y='petal_width',
        color='species',
        tooltip=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    )
    st.altair_chart(chart, use_container_width=True)
    
    st.markdown("### Sepal Length vs Sepal Width (Interactive)")
    chart2 = alt.Chart(df).mark_circle(size=60).encode(
        x='sepal_length',
        y='sepal_width',
        color='species',
        tooltip=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    ).interactive()
    st.altair_chart(chart2, use_container_width=True)

    # Slider for filtering rows
    st.subheader("Filter Rows by Sepal Length")
    min_sepal_length, max_sepal_length = st.slider(
        "Select range of Sepal Length", 
        float(df["sepal_length"].min()), 
        float(df["sepal_length"].max()), 
        (float(df["sepal_length"].min()), float(df["sepal_length"].max()))
    )
    filtered_df = df[(df["sepal_length"] >= min_sepal_length) & (df["sepal_length"] <= max_sepal_length)]
    st.write(filtered_df)

elif st.session_state.page_selection == 'data_cleaning':
    st.title('Data Cleaning / Pre-processing')
    st.write("This section will show the steps for cleaning and preprocessing the Iris dataset.")
    # Add your data cleaning logic here.

elif st.session_state.page_selection == 'machine_learning':
    st.title('Machine Learning')
    st.write("This section will show machine learning model training and evaluation.")
    st.subheader("Set Train/Test Split Ratio")
    train_test_ratio = st.slider("Train/Test Split", 0.1, 0.9, 0.8)
    st.write(f"Training with {train_test_ratio*100}% of the data")

elif st.session_state.page_selection == 'prediction':
    st.title('Prediction')
    st.subheader("Input Features for Prediction")
    sepal_length = st.slider("Sepal Length", float(df["sepal_length"].min()), float(df["sepal_length"].max()))
    sepal_width = st.slider("Sepal Width", float(df["sepal_width"].min()), float(df["sepal_width"].max()))
    petal_length = st.slider("Petal Length", float(df["petal_length"].min()), float(df["petal_length"].max()))
    petal_width = st.slider("Petal Width", float(df["petal_width"].min()), float(df["petal_width"].max()))
    
    st.write(f"Predicting with inputs: [{sepal_length}, {sepal_width}, {petal_length}, {petal_width}]")
    # Add your model prediction logic here.

elif st.session_state.page_selection == 'conclusion':
    st.title('Conclusion')
    st.markdown("The Iris classification app demonstrates how to build an interactive dashboard with Streamlit.")

# Footer Section
st.sidebar.subheader("Contact")
st.sidebar.text("Stéphane C. K. Tékouabou")
st.sidebar.text("ctekouaboukoumetio@gmail.com")
