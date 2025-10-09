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

loaded_model = pickle.load(open('trained_model.sav', 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

def diabetes_prediction(input_data_list):
    # Create a DataFrame from the input list
    input_data_data = pd.DataFrame([input_data_list], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'])
    
    # Scale the input data using the pre-trained scaler
    scaled_input_data = loaded_scaler.transform(input_data_data)
    
    # Convert scaled data to a NumPy array for prediction
    scaled_input_data_as_numpy_array = scaled_input_data.to_numpy()
    
    # Reshape the array
    input_data_reshaped = scaled_input_data_as_numpy_array.reshape(1, -1)
    
    # Get the prediction
    prediction = loaded_model.predict(input_data_reshaped)
    
    # Return the result
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():
    st.title('Diabetes Prediction app')
    
    # Use st.form to group the inputs and the button
    with st.form(key='diabetes_form'):
        Pregnancies = st.number_input('Number of Pregnancies', value=0.0)
        Glucose = st.number_input('Glucose value', value=0.0)
        BloodPressure = st.number_input('Blood Pressure value', value=0.0)
        SkinThickness = st.number_input('Skin Thickness value', value=0.0)
        Insulin = st.number_input('Insulin value', value=0.0)
        BMI = st.number_input('BMI value', value=0.0)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', value=0.0)
        
        # Using submit button for the result.
        submit_button = st.form_submit_button(label='Diabetes test result')
        
    # The code below this will only be executed after the form is submitted
    if submit_button:
        try:
            input_list = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction)
            ]
            
            # Check for invalid values like 0.0 or others if necessary
            
            if all(value == 0.0 for value in input_list):
                 st.error("Please fill in all the input fields with valid numbers.")
            else:
                 diagnosis = diabetes_prediction(input_list)
                 st.success(diagnosis)
        
        except ValueError:
            # This block will handle issues if input values aren't numbers
            st.error("Please ensure all inputs are valid numbers.")

            st.success(diagnosis)
        
        

if __name__ == '__main__':
    main()