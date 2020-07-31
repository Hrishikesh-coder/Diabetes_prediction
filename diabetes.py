import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

st.write('''
# DIABETES PREDICTION
''')


st.write("In this project, we shall try to predict if you have diabetes.")

df = pd.read_csv('kaggle_diabetes.csv')

df

X=df.drop('Outcome', axis=1)
y=df['Outcome']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, stratify=y)

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train,y_train)

def diabetes(preg, glucose, bp, skinThickness, insulin, BMI, DPF, age):
    predictions={0:'You dont have diabetes', 1:'You have diabetes'}
    prediction=classifier.predict([[preg, glucose, bp, skinThickness, insulin, BMI, DPF, age]])[0]
    return predictions[prediction]

preg = st.text_input("Enter preg :", 0)
glucose = st.text_input("Enter the glucose level:", 80)
bp = st.text_input("Enter your blood pressure: ",80)
sk = st.text_input("Enter your skin thickness:",30)
insulin = st.text_input("Enter your insulin:",120)
bmi = st.text_input("Enter your bmi:",35)
dpf = st.text_input("Enter your dpf : ")
age = st.text_input("Enter your age:")

st.write(diabetes(preg,glucose,bp,sk,insulin,bmi,dpf,age))