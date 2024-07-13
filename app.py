from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd

# Load and preprocess data
data = pd.read_csv("water_potability.csv").dropna()

# Upsample minority class
num = data.iloc[:, -1].value_counts()
minority_data = data[data.iloc[:, -1] == 1]
upsampled_data = pd.concat([minority_data.sample(num[0], replace=True, random_state=42), data[data.iloc[:, -1] == 0]])

# Preprocessing: Scaling and polynomial features
scaler = MinMaxScaler()
polynom = PolynomialFeatures(degree=3)

X = polynom.fit_transform(upsampled_data.iloc[:, :-1])
X_scaled = scaler.fit_transform(X)
y = upsampled_data.iloc[:, -1]

# Model training
model_rf = RandomForestClassifier()

model_rf.fit(X_scaled, y)


# Streamlit app
st.title('Water Potability Prediction')

# Sidebar for user input
st.sidebar.header('User Input Parameters')


# User inputs with min and max values in labels
ph = st.sidebar.number_input('ph (0.0 to 14.0)', value=6.704635, min_value=0.0, max_value=14.0)
Hardness = st.sidebar.number_input('Hardness (0.0 to 500.0)', value=230.766940, min_value=0.0, max_value=500.0)
Solids = st.sidebar.number_input('Solids (0.0 to 50000.0)', value=9727.761716, min_value=0.0, max_value=50000.0)
Chloramines = st.sidebar.number_input('Chloramines (0.0 to 15.0)', value=5.943695, min_value=0.0, max_value=15.0)
Sulfate = st.sidebar.number_input('Sulfate (0.0 to 1000.0)', value=223.235816, min_value=0.0, max_value=1000.0)
Conductivity = st.sidebar.number_input('Conductivity (0.0 to 2000.0)', value=405.761571, min_value=0.0, max_value=2000.0)
Organic_carbon = st.sidebar.number_input('Organic Carbon (0.0 to 30.0)', value=12.826509, min_value=0.0, max_value=30.0)
Trihalomethanes = st.sidebar.number_input('Trihalomethanes (0.0 to 200.0)', value=74.385199, min_value=0.0, max_value=200.0)
Turbidity = st.sidebar.number_input('Turbidity (0.0 to 10.0)', value=3.422179, min_value=0.0, max_value=10.0)





# Prediction
if st.button('Predict'):
    # Preprocess user input
    data = {
        'ph': ph,
        'Hardness': Hardness,
        'Solids': Solids,
        'Chloramines': Chloramines,
        'Sulfate': Sulfate,
        'Conductivity': Conductivity,
        'Organic_carbon': Organic_carbon,
        'Trihalomethanes': Trihalomethanes,
        'Turbidity': Turbidity
    }
    features = pd.DataFrame(data, index=[0])

    input_poly = polynom.transform(features)
    input_scaled = scaler.transform(input_poly)
    
    # Make prediction
    prediction = model_rf.predict(input_scaled)
    prediction_proba = model_rf.predict_proba(input_scaled)
    
    # Output prediction
    st.subheader('Prediction')
    potability = 'Potable' if prediction[0] == 1 else 'Not Potable'
    st.write(potability)
    
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
