# Disease Prediction Models
In this project, I used the dataset hearts_2020_cleaned.csv to build Logistic Regression and Decision Tree models that predict whether an individual has heart disease, kidney disease, or skin cancer based on various health and lifestyle features.

## Dataset Features
The dataset includes the following columns:
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

## Data Preparation:
1. Binary Encoding: Columns with "Yes"/"No" values were converted to binary (1/0) using **LabelEncoder**:
```
from sklearn.preprocessing import LabelEncoder

cols = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity',
       'Asthma', 'KidneyDisease', 'SkinCancer']

for col in cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[[col]])
```
2. Ordinal Encoding: The **GenHealth** column was encoded to reflect the order of ratings:
```
from sklearn.preprocessing import OrdinalEncoder

rating = ['Poor', 'Fair', "Excellent", "Good", "Very good"]
ordinal_encoder = OrdinalEncoder(categories=[rating])
df["GenHealth"] = ordinal_encoder.fit_transform(df[['GenHealth']])
```
3. One-Hot Encoding: Remaining categorical columns were encoded using one-hot encoding:

```
df = pd.get_dummies(df,columns=['AgeCategory', 'Race', 'Diabetic'], drop_first=True,dtype=int)
```
After preprocessing, the final dataset contained 35 features.

## Logistic Regression
Three separate logistic regression models were built for each of the three diseases:

### Heart Disease
<img src='images/HeartDisease_Logistic.png'>
<img src= 'images/heart_cm_logistic.png' width=400px height = 400px>

### Kidney Disease
<img src='images/KidneyDisease_Logistic.png'>
<img src= 'images/kidney_cm_logistic.png' width=400px height = 400px>

### Skin Cancer
<img src='images/SkinCancer_Logistic.png'>
<img src= 'images/skin_cm_logistic.png' width=400px height = 400px>

Across all three models, there was no sign of overfitting, as training and testing accuracies were consistently similar. The logistic regression models demonstrated relatively high recall scores, indicating strong performance in identifying positive cases. Notably, the skin cancer model achieved a **recall of 0.83**, meaning it **correctly identified 83% of individuals who actually had skin cancer**.

However, precision was low across all models, suggesting that a significant number of the positive predictions were incorrect. As a result, the F1-scores ranged between 0.15 and 0.35, reflecting an imbalance between precision and recall. Among the three models, the **heart disease model achieved the highest AUC**, indicating it had the best overall ability to distinguish between classes.

The imbalance between recall and precision may be due to the class imbalance in the dataset. This is expected, as the majority of individuals do not have heart disease, kidney disease, or skin cancer. To address this issue, sampling techniques such as **SMOTE, ADASYN, RandomOverSampler, and RandomUnderSampler** can be applied. However, it’s important to note that using these techniques can increase the risk of overfitting.


## Decision Trees 
Decision Tree classifiers were also trained to predict the same three conditions. Below are their evaluation results:

### Heart Disease
<img src="images/Heart_cm_decisiontree.png" width=400px height = 400px>
<img src="images/heart_decisiontree.png" width=400px height = 400px>

### Kidney Disease
<img src="images/kidney_cm_decisiontree.png" width=400px height = 400px>
<img src="images/kidney_decisontree.png" width=400px height = 400px>

### Skin Cancer
<img src="images/skin_cm_decisiontree.png" width=400px height = 400px>
<img src="images/skin_decisontree.png" width=400px height = 400px>

All three **Decision Tree models** demonstrated significantly lower precision and recall scores compared to their corresponding logistic regression models.  For instance, the recall for detecting heart disease in the logistic regression model was **76%**, while the decision tree model achieved only **54%**. This performance gap may be attributed to overfitting, as suggested by the noticeable difference between training and testing accuracy.
To help reduce overfitting, the hyperparameter ccp_alpha was tuned and set to 0.002 to enable cost-complexity pruning, to help choose an optimal ccp_alpha value cross validation could be used however, it took a while to do:
```
# Decision Tree
clf = DecisionTreeClassifier(ccp_alpha=0.002, class_weight='balanced', random_state=42)
clf.fit(X_train_res, y_train_res)
```
Although cross-validation could have been used to select the optimal ccp_alpha value, the process was computationally intensive and time-consuming for this dataset.


To address class imbalance in the dataset, SMOTE (Synthetic Minority Oversampling Technique) was applied during training:

```
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```
Despite these efforts to improve performance, through both pruning and resampling, the Decision Tree models did not outperform the logistic regression models. This may be due to other factors such as pruning, tree depth, or the choice of splitting criteria (e.g., Gini impurity vs. entropy). Further tuning and model experimentation could potentially improve results.
