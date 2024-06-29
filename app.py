# %%
import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load the dataset
data = pd.read_csv("water_potability.csv")

# Split the dataset into training and testing sets
train = data.sample(frac=0.8, random_state=42)
test = data.drop(train.index)

# Initialize SimpleImputer and StandardScaler
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# Fit imputer and scaler with training data
imputer.fit(train.iloc[:, :-1])
scaler.fit(train.iloc[:, :-1])

# Initialize and train the model
model_random = XGBClassifier(random_state=42)
model_random.fit(train.iloc[:, :-1], train.iloc[:, -1])

# Streamlit app
st.title('Water Potability Prediction')

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

# Add a button for prediction
if st.button('Predict'):
    # Preprocess user input
    input_df_transformed = pd.DataFrame(imputer.transform(input_df), columns=train.columns[:-1])
    input_df_transformed = pd.DataFrame(scaler.transform(input_df_transformed), columns=train.columns[:-1])

    # Make prediction
    prediction = model_random.predict(input_df_transformed)
    prediction_proba = model_random.predict_proba(input_df_transformed)

    # Output prediction
    st.subheader('Prediction')
    potability = 'Potable' if prediction[0] == 1 else 'Not Potable'
    st.write(potability)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)


# %%



