import pandas as pd
import numpy as np


stroke = pd.read_csv('processed.csv')

stroke = stroke.drop(['smoking_status'], axis = 1)
stroke = stroke.drop(['Residence_type'], axis = 1)
#stroke['gender'] = stroke['gender'].replace({'Other': np.nan})

Y = stroke['stroke']
X = stroke.drop(['stroke'], axis = 1)

stroke.to_csv('processed.csv', index=False)