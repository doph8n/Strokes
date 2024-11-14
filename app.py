import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

stroke = pd.read_csv('processed.csv')



def graphs_page():
    st.title("Graphs")
    graph_type = st.selectbox("Select a graph", ["Correlation Heatmap", "Gender", "Age", "Smoking Status", "Hypertension", "Heart Disease", "Work Type", "BMI"])

    if graph_type == "Correlation Heatmap":
        plt.figure(figsize=(10, 8))
        correlation_matrix = stroke.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, cbar_kws={'shrink': .8})
        plt.title('Correlation Heatmap of Health Features', fontsize=16)
        plt.xticks(rotation=45, fontsize=10, ha='right')
        plt.yticks(fontsize=10)
        st.pyplot(plt.gcf())

    if graph_type == "Gender":
        plt.figure(figsize=(8,6))
        gender_prop = pd.crosstab(index=stroke['gender'], columns=stroke['stroke'], normalize="index")
        gender_prop.plot(kind='bar', stacked=True, colormap='tab10', figsize=(10, 6))
        plt.legend(loc="lower left", ncol=2)
        plt.title('Proportion of Stroke Occurrences by Gender')
        plt.xlabel("Gender")
        plt.ylabel("Proportion")
        plt.xticks([0, 1, 2], ['Male', 'Female', "Other"], rotation=0)
        st.pyplot(plt.gcf()) 

    if graph_type == "Age":
        stroke['age_range'] = pd.cut(x = stroke['age'], bins = [0, 18, 55, 75, 100], 
                             labels = ['children', 'adults', 'senior', 'elderly'])
        plt.figure(figsize=(8,6))
        age_prop = pd.crosstab(index=stroke['age_range'], columns=stroke['stroke'], normalize="index")
        age_prop.plot(kind='bar', stacked=True, colormap='tab10', figsize=(10, 6))
        plt.legend(loc="lower left", ncol=2)
        plt.title('Proportion of Stroke Occurrences by Age Range')
        plt.xlabel("Age")
        plt.ylabel("Proportion")
        st.pyplot(plt.gcf()) 

    elif graph_type == "Smoking Status":
        plt.figure(figsize=(8,6))
        smoke_prop = pd.crosstab(index=stroke['smoking_status'], columns=stroke['stroke'], normalize="index")
        smoke_prop.plot(kind='bar', stacked=True, colormap='tab10', figsize=(10, 6))
        plt.legend(loc="lower left", ncol=2)
        plt.title('Proportion of Stroke Occurrences by Smoking Status')
        plt.xlabel("Smoking Status")
        plt.ylabel("Proportion")
        plt.xticks([0, 1, 2, 3], ['Never Smokes', 'Formerly Smokes', "Smokes", "Unknown"], rotation=0)
        st.pyplot(plt.gcf())  
    
    elif graph_type == "Hypertension":
        plt.figure(figsize=(8,6))
        hypertension_prop = pd.crosstab(index=stroke['hypertension'], columns=stroke['stroke'], normalize="index")
        hypertension_prop.plot(kind='bar', stacked=True, colormap='tab10', figsize=(8, 6))     
        plt.title('Proportion of Stroke Occurrences by Hypertension Status')
        plt.xlabel('Hypertension')
        plt.ylabel('Proportion')
        plt.legend(title='Stroke', loc="lower left", ncol=2)
        plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
        st.pyplot(plt.gcf())  
    
    elif graph_type == "Heart Disease":
        plt.figure(figsize=(8,6))
        heartdisease_prop = pd.crosstab(index=stroke['heart_disease'], columns=stroke['stroke'], normalize="index")
        heartdisease_prop.plot(kind='bar', stacked=True, colormap='tab10', figsize=(8, 6))
        plt.title('Proportion of Stroke Occurrences by Heart Disease Status')
        plt.xlabel('Heart Disease')
        plt.ylabel('Proportion')
        plt.legend(title='Stroke', labels=['No Stroke', 'Stroke'], loc="lower left", ncol=2)
        plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
        st.pyplot(plt.gcf())  

    elif graph_type == "Work Type":
        plt.figure(figsize=(8,6))
        work_prop = pd.crosstab(index=stroke['work_type'], columns=stroke['stroke'], normalize="index")
        work_prop.plot(kind='bar', stacked=True, colormap='tab10', figsize=(10, 6))
        plt.title('Proportion of Stroke Occurences by Work Type', fontsize=16)
        plt.legend(loc="lower left", ncol=2)
        plt.xlabel("Work Type")
        plt.ylabel("Proportion")
        plt.xticks([0, 1, 2, 3, 4], ['Child', 'Never Worked', "Self-Employed", "Private", "Government-Employed" ], rotation=0)
        st.pyplot(plt.gcf())  

    elif graph_type == "BMI":
        stroke['bmi_range'] = pd.cut(x=stroke['bmi'], bins = [0,18,25,30,60],
                          labels= ['underweight', 'healthy', 'overweight', 'obese'])
        plt.figure(figsize=(8,6))
        bmi_prop = pd.crosstab(index=stroke['bmi_range'], columns=stroke['stroke'], normalize="index")
        bmi_prop.plot(kind='bar', stacked=True, colormap='tab10', figsize=(10, 6))
        plt.title('Proportion of Stroke Occurences by BMI', fontsize=16)
        plt.legend(loc="lower left", ncol=2)
        plt.xlabel("BMI")
        plt.ylabel("Proportion")
        st.pyplot(plt.gcf())  



# gender,age,hypertension,heart_disease,ever_married,work_type,avg_glucose_level,bmi,stroke

def main_page():
    st.title("Stroke Predictor")
    with open("stroke_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    st.write("**Enter information below:**")

    gender_prompt = st.radio("What's your gender?", ['Male', 'Female', "Other"], index=0)
    gender = 0 if gender_prompt == "Male" else 1 

    age = st.number_input("What's your age?", value=None, placeholder="Type a number...")
    
    hypertension_prompt = st.radio("Do you have hypertension?", ["Yes", "No"], index=0)
    hypertension = 0 if hypertension_prompt == "No" else 1

    heart_disease_prompt = st.radio("Do you have heart disease?", ["Yes", "No"], index=0)
    heart_disease = 0 if heart_disease_prompt == "No" else 1

    ever_married_prompt = st.radio("Have you ever been married?", ["Yes", "No"], index=0)
    ever_married = 0 if ever_married_prompt == "No" else 1

    work_type_prompt = st.radio("What is your work type?", ['Child', 'Never Worked', "Self-Employed", "Private", "Government-Employed"], index=0)
    work_type_encoded = {"Child": 0, "Never Worked": 1, "Self-Employed": 2, "Private": 3, "Goverment-Employed": 4}
    work_type = work_type_encoded[work_type_prompt]

    avg_glucose_level = st.number_input("What's your average glucose level?", value=None, placeholder="Type a number...")
    bmi = st.number_input("What's your BMI?", value=None, placeholder="Type a number...")
        
    
    predicted_stroke = None

    if st.button("Predict!"):
        # Perform prediction
        features = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, avg_glucose_level, bmi]])
        predicted_stroke = model.predict(features)[0]
        
        # Display prediction result
        st.write(f"**Prediction:** {'High Risk' if predicted_stroke == 1 else 'Low Risk'}")
    
    
#Side Bar stuff
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home Page", "Graphs Page"])

if page == "Home Page":
    main_page()
elif page == "Graphs Page":
    graphs_page()
    