import os
import mglearn
import pandas as pd
adult_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
data = pd.read_csv(
    adult_path, header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race','gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
)

data = data[['age','workclass','education','gender','hours-per-week',
             'occupation','income']]

##print(data.gender.value_counts())

data_dummies = pd.get_dummies(data)
print(data_dummies.columns)



