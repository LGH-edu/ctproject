```python
# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# Set page config
st.set_page_config(page_title="Leaf Classification App", layout="wide")

# Create sample data
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'name': [f'Leaf_{i}' for i in range(n_samples)],
        'leaf_length': np.random.uniform(5, 15, n_samples),
        'leaf_width': np.random.uniform(2, 8, n_samples),
        'kind': np.random.choice(['parallel', 'net'], n_samples)
    }
    return pd.DataFrame(data)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = create_sample_data()
if 'model' not in st.session_state:
    st.session_state.model = None

# Main app
def main():
    st.title("🍃 Leaf Classification System")
    
    # Sidebar menu
    with st.sidebar:
        st.header("Menu")
        page = st.radio("Select a page:", 
                       ["Tutorial", "Data Upload", "Model Training", "Prediction"])
    
    if page == "Tutorial":
        show_tutorial()
    elif page == "Data Upload":
        data_upload()
    elif page == "Model Training":
        model_training()
    elif page == "Prediction":
        make_prediction()

def show_tutorial():
    st.header("How to Use This App")
    st.write("""
    1. Start by uploading your leaf data or use our sample dataset
    2. Train the model using the provided data
    3. Input leaf measurements to get predictions
    """)
    
    # Show sample data
    st.subheader("Sample Data Preview")
    st.dataframe(st.session_state.data.head())

def data_upload():
    st.header("Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload your leaf data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error uploading file: {e}")
    
    with col2:
        if st.button("Use Sample Data"):
            st.session_state.data = create_sample_data()
            st.success("Sample data loaded!")
    
    # Display data visualization
    if st.session_state.data is not None:
        st.subheader("Data Visualization")
        fig = px.scatter(st.session_state.data, 
                        x='leaf_length', 
                        y='leaf_width',
                        color='kind',
                        title='Leaf Measurements Distribution')
        st.plotly_chart(fig)

def model_training():
    st.header("Model Training")
    
    if st.button("Train Model"):
        try:
            # Prepare data
            X = st.session_state.data[['leaf_length', 'leaf_width']]
            le = LabelEncoder()
            y = le.fit_transform(st.session_state.data['kind'])
            
            # Train model
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            
            st.session_state.model = model
            st.session_state.label_encoder = le
            
            st.success("Model trained successfully!")
            
            # Show model accuracy
            score = model.score(X, y)
            st.metric("Model Accuracy", f"{score:.2%}")
            
        except Exception as e:
            st.error(f"Error training model: {e}")

def make_prediction():
    st.header("Leaf Classification")
    
    if st.session_state.model is None:
        st.warning("Please train the model first!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        length = st.number_input("Enter leaf length (cm)", 0.0, 20.0, 10.0)
    with col2:
        width = st.number_input("Enter leaf width (cm)", 0.0, 10.0, 5.0)
    
    if st.button("Classify"):
        try:
            prediction = st.session_state.model.predict([[length, width]])
            predicted_class = st.session_state.label_encoder.inverse_transform(prediction)[0]
            
            st.success(f"Predicted leaf type: {predicted_class}")
            
            # Show prediction visualization
            fig = px.scatter(st.session_state.data, 
                           x='leaf_length', 
                           y='leaf_width',
                           color='kind',
                           title='Prediction Visualization')
            
            # Add prediction point
            fig.add_scatter(x=[length], y=[width], 
                          mode='markers',
                          marker=dict(size=15, symbol='star', color='yellow'),
                          name='Your Input')
            
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
```