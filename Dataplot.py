import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

stroke = pd.read_csv('processed.csv')

stroke['age_range'] = pd.cut(x = stroke['age'], bins = [0, 18, 55, 75, 100], 
                             labels = ['children', 'adults', 'senior', 'elderly'])

stroke['glucose_range'] = pd.cut(x = stroke['avg_glucose_level'], bins = [0, 18, 55, 75, 100], 
                             labels = ['children', 'adults', 'senior', 'elderly'])

# Use countplot to examine relationships between variables and target (stroke)
#sns.countplot(data=stroke, x='age_range', hue = 'stroke')

#sns.countplot(data=stroke, x='heart_disease', hue = 'stroke')
#plt.show()

stroke['bmi_range'] = pd.cut(x=stroke['bmi'], bins = [0,18,25,30,60],
                          labels= ['underweight', 'healthy', 'overweight', 'obese'])

sns.countplot(data=stroke, x='gender', hue = 'stroke')
plt.show()


smoke_prop = pd.crosstab(index=stroke['smoking_status'],
                             columns=stroke['stroke'],
                             normalize="index")
smoke_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))
plt.legend(loc="lower left", ncol=2)
plt.xlabel("Smoking Status")
plt.ylabel("Proportion")
plt.show()

bmi_prop = pd.crosstab(index=stroke['bmi_range'],
                             columns=stroke['stroke'],
                             normalize="index")
bmi_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))
plt.legend(loc="lower left", ncol=2)
plt.xlabel("BMI")
plt.ylabel("Proportion")
plt.show()

heartdisease_prop = pd.crosstab(index=stroke['heart_disease'], columns=stroke['stroke'], normalize="index")
heartdisease_prop.plot(kind='bar', stacked=True, colormap='tab10', figsize=(8, 6))
plt.title('Proportion of Stroke Occurrences by Heart Disease Status')
plt.xlabel('Heart Disease')
plt.ylabel('Proportion')
plt.legend(title='Stroke', labels=['No Stroke', 'Stroke'], loc="lower left", ncol=2)
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
plt.show()


gender_prop = pd.crosstab(index=stroke['gender'],
                             columns=stroke['stroke'],
                             normalize="index")
gender_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))
plt.legend(loc="lower left", ncol=2)
plt.xlabel("Gender")
plt.ylabel("Proportion")
plt.show()

hypertension_prop = pd.crosstab(index=stroke['hypertension'], columns=stroke['stroke'], normalize="index")
hypertension_prop.plot(kind='bar', stacked=True, colormap='tab10', figsize=(8, 6))
plt.title('Proportion of Stroke Occurrences by Hypertension Status')
plt.xlabel('Hypertension')
plt.ylabel('Proportion')
plt.legend(title='Stroke', loc="lower left", ncol=2)
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
plt.show()

work_prop = pd.crosstab(index=stroke['work_type'],
                             columns=stroke['stroke'],
                             normalize="index")
work_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))
plt.legend(loc="lower left", ncol=2)
plt.xlabel("Work Type")
plt.ylabel("Proportion")
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = stroke.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, cbar_kws={'shrink': .8})
plt.title('Correlation Heatmap of Health Features', fontsize=16)
plt.xticks(rotation=45, fontsize=10, ha='right')
plt.yticks(fontsize=10)
plt.show()
