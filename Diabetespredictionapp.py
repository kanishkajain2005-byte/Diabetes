#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 19:57:15 2025

@author: Jaishreenirmala
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('/Users/Jaishreenirmala/Desktop/Diabetes prediction/trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    #input_data = (1,189,60,23,846,30.1,0.398)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0] == 0):
      return ('The person is not diabetic')
    else:
      return ('The person is diabetic')
  
def main():
    st.title('Diabetes Prediction app')
    
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose value')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    diognosis = ''
    
    if st.button('Diabetes test result'):
        
        # Use a try-except block to handle cases where input fields are empty or invalid
        try:
            # All inputs are converted to floats here.
            # If any of them are empty strings, a ValueError will be raised.
            input_list = [
                float(BMI), float(Glucose), float(SkinThickness), 
                float(Insulin), float(BloodPressure), 
                float(Pregnancies), float(DiabetesPedigreeFunction)
            ]
            
            # Pass the list of floats to the prediction function
            diagnosis = diabetes_prediction(input_list)
            
            st.success(diagnosis)
        except ValueError:  
            # This block will run if any of the float() conversions fail,
            # which happens if the input string is empty ("") or not a number.
            st.error("Please fill in all the input fields with valid numbers.")


if __name__ == '__main__'  :
    main()