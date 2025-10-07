# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

loaded_model = pickle.load(open('/Users/Jaishreenirmala/Desktop/Diabetes prediction/trained_model.sav', 'rb'))
loaded_scaler = pickle.load(open('/Users/Jaishreenirmala/Desktop/Diabetes prediction/scaler.pkl', 'rb'))

# taking the input for the prediction.
input_data = pd.DataFrame([input_data], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'])
# now scaling the input taken.
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