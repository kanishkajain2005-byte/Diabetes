#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 19:57:15 2025

@author: Jaishreenirmala
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd

loaded_model = pickle.load(open('/Users/Jaishreenirmala/Desktop/Diabetes prediction/trained_model.sav', 'rb'))
loaded_scaler = pickle.load(open('/Users/Jaishreenirmala/Desktop/Diabetes prediction/scaler.pkl', 'rb'))

def diabetes_prediction(input_data):
    # taking the input for the prediction.
    input_data = pd.DataFrame([input_data], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'])
    # scaling the taken input.
    scaled_input_data = loaded_scaler.transform(input_data)
    # now converting the input into a array.
    scaled_input_data_as_numpy_array = np.asarray(scaled_input_data)
    # reshaping the array.
    scaled_input_data_reshaped = scaled_input_data_as_numpy_array.reshape(1,-1)
    # now predicting.
    prediction = loaded_model.predict(scaled_input_data_reshaped)
    # putting the condition for prediction.
    print(prediction)
    if (prediction[0] == 0):
      return ('The person is not diabetic')
    else:
      return ('The person is diabetic')
  
def main():
    # creating the title of the app.
    st.title('Diabetes Prediction app')
    
    # Taking the inputs of the following values.
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose value')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    diognosis = ''
    
    if st.button('Diabetes test result'):
        
        
        try:
            
            input_list = [
                float(BMI), float(Glucose), float(SkinThickness), 
                float(Insulin), float(BloodPressure), 
                float(Pregnancies), float(DiabetesPedigreeFunction)
            ]
            
            #
            diagnosis = diabetes_prediction(input_list)
            
            st.success(diagnosis)
            # putting the condition for empty entries.
        except ValueError:  
            
            st.error("Please fill in all the input fields with valid numbers.")


if __name__ == '__main__'  :
    main()