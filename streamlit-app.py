# Import necessary libraries
import streamlit as st
import pandas as pd
import altair as alt
import sklearn
import joblib  # For saving and loading the model


# Configure the Streamlit app
st.set_page_config(
    page_title="Iris Classification Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.themes.enable("dark")  # Use a dark theme for charts

# Load the dataset
@st.cache_data
def load_data(file_path):
    """Load and cache the Iris dataset."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'iris.csv' is in the correct directory.")
        return pd.DataFrame()

# Dataset loading
df = load_data('iris.csv')

# Initialize page selection state
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'

# Function to change the selected page
def set_page(page):
    st.session_state.page_selection = page

# Sidebar Navigation
with st.sidebar:
    st.title("🌸 Iris Classification")
    st.subheader("Navigation")
    pages = {
        "About": "about",
        "Dataset Overview": "dataset",
        "Exploratory Data Analysis": "eda",
        "Data Preprocessing": "preprocessing",
        "Machine Learning": "ml",
        "Prediction": "prediction",
        "Conclusion": "conclusion",
    }
    for page, page_key in pages.items():
        st.button(page, on_click=set_page, args=(page_key,))

    # Project Details
    st.subheader("Project Details")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")

# Main Content Area
st.title("Iris Classification Dashboard")

if st.session_state.page_selection == 'about':
    st.header("About")
    st.write("""
    Welcome to the *Iris Classification Dashboard*!  
    This interactive application allows you to explore the famous Iris dataset, conduct Exploratory Data Analysis (EDA), 
    preprocess the data, train machine learning models, and make predictions.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_versicolor_3.jpg", caption="Iris Flower", use_column_width=True)
    st.markdown("""
    ### Features:
    - Dataset visualization and analysis.
    - Data cleaning and preprocessing.
    - Machine learning model training and evaluation.
    - Real-time predictions with trained models.
    """)

elif st.session_state.page_selection == 'dataset':
    st.header("Dataset Overview")
    if not df.empty:
        st.subheader("First 10 Rows of the Dataset")
        st.write(df.head(10))
        st.subheader("Dataset Summary")
        st.write(df.describe())
        st.subheader("Column Information")
        st.write(df.info())
    else:
        st.error("Dataset is not loaded. Please ensure 'iris.csv' is available.")

elif st.session_state.page_selection == 'eda':
    st.header("Exploratory Data Analysis (EDA)")
    if not df.empty:
        st.subheader("Scatter Plot: Petal Dimensions")
        scatter_plot = alt.Chart(df).mark_circle(size=60).encode(
            x='petal_length',
            y='petal_width',
            color='species',
            tooltip=['petal_length', 'petal_width', 'species']
        ).interactive()
        st.altair_chart(scatter_plot, use_container_width=True)

        st.subheader("Scatter Plot: Sepal Dimensions")
        scatter_plot_2 = alt.Chart(df).mark_circle(size=60).encode(
            x='sepal_length',
            y='sepal_width',
            color='species',
            tooltip=['sepal_length', 'sepal_width', 'species']
        ).interactive()
        st.altair_chart(scatter_plot_2, use_container_width=True)
    else:
        st.error("Dataset is not loaded. Please ensure 'iris.csv' is available.")

# Preprocessing
elif st.session_state.page_selection == 'preprocessing':
    st.header("Data Preprocessing")
    st.write("""
    Data preprocessing is a crucial step in any machine learning workflow. 
    It ensures the data is clean, structured, and ready for model training.
    """)

    # Step 1: Check for Missing Values
    st.subheader("1. Handle Missing Values")
    if st.checkbox("Check for missing values"):
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.write("### Missing Values Summary:")
            st.write(missing_values)
            if st.button("Fill Missing Values with Mean"):
                df.fillna(df.mean(), inplace=True)
                st.success("Missing values filled with column mean.")
        else:
            st.success("No missing values detected!")

    # Step 2: Encode Categorical Variables
    st.subheader("2. Encode Categorical Variables")
    if st.checkbox("Convert categorical variables into numerical values"):
        if 'species' in df.columns:
            df['species_encoded'] = df['species'].astype('category').cat.codes
            st.write("### Updated Dataset:")
            st.write(df.head())
            st.success("Categorical variable 'species' encoded successfully.")
        else:
            st.warning("No categorical variables to encode.")

    # Step 3: Feature Scaling
    st.subheader("3. Standardize Numerical Features")
    if st.checkbox("Standardize features to have mean=0 and variance=1"):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        st.write("### Standardized Dataset:")
        st.write(df.head())
        st.success("Numerical features standardized successfully.")

    # Step 4: Save Processed Dataset
    st.subheader("4. Save Processed Dataset")
    if st.button("Save Processed Data"):
        df.to_csv("processed_iris.csv", index=False)
        st.success("Processed dataset saved as 'processed_iris.csv'.")

# Machine Learning
elif st.session_state.page_selection == 'ml':
    st.header("Machine Learning")
    st.subheader("Train and Evaluate Models")
    
    # Select model
    st.write("### Step 1: Choose a Model")
    model_option = st.selectbox(
        "Select a machine learning model",
        ("Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine")
    )
    
    # Training process
    st.write("### Step 2: Train the Model")
    test_size = st.slider("Select the test size (%)", min_value=10, max_value=50, step=5, value=30)
    if st.button("Train Model"):
        # Import necessary libraries
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        # Prepare the data
        X = df.drop(columns=["species"])  # Features
        y = df["species"]  # Target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # Train selected model
        if model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
        elif model_option == "Support Vector Machine":
            model = SVC()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save the model
        joblib.dump(model, 'iris_model.pkl')

        st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")

# Prediction
elif st.session_state.page_selection == 'prediction':
    st.header("Prediction")
    st.subheader("Make Predictions Using Trained Models")
    
    # Input features
    st.write("### Enter Feature Values")
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

    # Predict button
    if st.button("Predict"):
        # Load the trained model
        try:
            model = joblib.load('iris_model.pkl')
            input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
            prediction = model.predict(input_data)
            st.success(f"The predicted species is: {prediction[0]}")
        except FileNotFoundError:
            st.error("Please train a model in the 'Machine Learning' section first!")

elif st.session_state.page_selection == 'conclusion':
    st.header("Conclusion")
    st.write("""
    ### Key Takeaways:
    - The Iris dataset demonstrates the power of data analysis and machine learning.
    - Classification models can effectively distinguish between different species of flowers.
    - Interactive dashboards provide a great way to present insights.
    """)

# Footer
st.markdown("---")
st.markdown("App built with ❤ by Stéphane C. K. Tékouabou")
