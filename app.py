import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

model = load_model('model.h5')



with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


st.title("Customer Churn Prediction")
geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox("Gender",label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('balance')
credit_score=st.number_input('Credit_score')
Estimated_Salary=st.number_input('Estimated_Salary')
tenure = st.slider('Tenure',0,10)
num_of_products=st.slider('Number Of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [Estimated_Salary]

}
)

geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


data=pd.concat([data.reset_index(drop=True),geo_encoded_df],axis=1)

scaled=scaler.transform(data)

prediction=model.predict(scaled)
prediction_prob=prediction[0][0]

if prediction_prob > 0.5:
    st.write("The costomer is likely to churn")
else:
    st.write("the costomer is not likely to churn")