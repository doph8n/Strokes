import pandas as pd
import numpy as np

# pandas - cleaning data --------------------------------------------------------------------------------------------------------

# Turns the CSV into a dataframe named stroke
stroke = pd.read_csv('healthcare-dataset-stroke-data - healthcare-dataset-stroke-data.csv')

# Drops IDs and Stroke as it is non important for the ML
stroke = stroke.drop(['id'], axis = 1)

pd.set_option('future.no_silent_downcasting', True)

# Removing N/A from the BMI data and inputing the mean
Bmi_Mean = stroke['bmi'].mean()
stroke.replace(['N/A'], np.nan, inplace=False)
stroke['bmi'] = stroke['bmi'].fillna(Bmi_Mean, inplace = False)

#stroke['work_type'] = pd.Categorical(stroke['work_type'])
#print(stroke['work_type'].code)

# Turning gender into a number, 0 = male, 1 = female
stroke['gender'] = stroke['gender'].replace({'Male': 0, 'Female': 1, 'Other': 2})

# Turning ever_married into a number, 0 = Yes , 1 = No
stroke['ever_married'] = stroke['ever_married'].replace({'Yes': 0, 'No': 1})

# Turning work_type into a number, 0 = child , 1 = never_worked 2 = self-employed, 3 = private, 4 = goverment job
stroke['work_type'] = stroke['work_type'].replace({'children': 0, 'Never_worked': 1, 'Self-employed': 2, 'Private': 3, 'Govt_job': 4})
 
# Turning residence_type into a number, 0 = urban, 1 = rural
stroke['Residence_type'] = stroke['Residence_type'].replace({'Urban': 0, 'Rural': 1})

# Turning smoking_status into a number, 0 = never smoked, 1 = unknown, 2 = formerly smokes, 3 = smokes
stroke['smoking_status'] = stroke['smoking_status'].replace({'never smoked': 0, 'Unknown': 1, 'formerly smoked': 2, 'smokes': 3})

stroke.to_csv('processed.csv', index=False)
