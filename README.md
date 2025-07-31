# Disease Prediction Models
In this project, I used hearts_2020_cleaned,csv to create Logistic Regression and Decision Tree Models to predict whether someone has kidney disease, heart disease, or skin cancer based on the features in the dataset. Columns in the dataset include:
- HeartDisease 
- BMI
- Smoking
- AlcoholDrinking
- Stroke
- PhysicalHealth
- MentalHealth
- DiffWalking
- Sex
- AgeCategory
- Race
- Diabetic
- PhysicalActivity
- GenHealth
- SleepTime
- Asthma
- KidneyDisease
- SkinCancer

The first step was to prepare the data and clean the data by converting categorical columns that had "yes" or "no" as values into 1 and 0 by using *LabelEncoder* in Python:
```
from sklearn.preprocessing import LabelEncoder

cols = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity',
       'Asthma', 'KidneyDisease', 'SkinCancer']

for col in cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[[col]])
```
For columns that had ordinal categorical values **OrdinalEncoder** was applied by using the following code:
```
from sklearn.preprocessing import OrdinalEncoder

rating = ['Poor', 'Fair', "Excellent", "Good", "Very good"]
ordinal_encoder = OrdinalEncoder(categories=[rating])
df["GenHealth"] = ordinal_encoder.fit_transform(df[['GenHealth']])
```
All other categorical columns were converted to numeric columns using one-hot encoding: 

```
df = pd.get_dummies(df,columns=['AgeCategory', 'Race', 'Diabetic'], drop_first=True,dtype=int)
```
After running one-hot encoding and converting categorical columns to binary columns there were 35 total columns in the dataset.

### Logistic Regression
I ran three separate logistic regression models where each model had a separate response variable: Heart Disease, Kidney Disease, Skin Cancer.






