from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from xgboost import XGBClassifier
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
model_xgb = XGBClassifier(random_state=42, n_estimators=300)
model_xgb.fit(X_scaled, y)

# Streamlit app
st.title('Water Potability Prediction')

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    ph = st.sidebar.number_input('ph', value=6.704635)
    Hardness = st.sidebar.number_input('Hardness', value=230.766940)
    Solids = st.sidebar.number_input('Solids', value=9727.761716)
    Chloramines = st.sidebar.number_input('Chloramines', value=5.943695)
    Sulfate = st.sidebar.number_input('Sulfate', value=223.235816)
    Conductivity = st.sidebar.number_input('Conductivity', value=405.761571)
    Organic_carbon = st.sidebar.number_input('Organic_carbon', value=12.826509)
    Trihalomethanes = st.sidebar.number_input('Trihalomethanes', value=74.385199)
    Turbidity = st.sidebar.number_input('Turbidity', value=3.422179)
    
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
    return features

input_df = user_input_features()

# Display user input
st.subheader('User Input parameters')
st.write(input_df)

# Prediction
if st.button('Predict'):
    # Preprocess user input
    input_poly = polynom.transform(input_df)
    input_scaled = scaler.transform(input_poly)
    
    # Make prediction
    prediction = model_xgb.predict(input_scaled)
    prediction_proba = model_xgb.predict_proba(input_scaled)
    
    # Output prediction
    st.subheader('Prediction')
    potability = 'Potable' if prediction[0] == 1 else 'Not Potable'
    st.write(potability)
    
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
