import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib

card_df_cleaned=pd.read_csv("cleaned data/cards_data_cleaned.csv")
trans_df_cleaned=pd.read_csv("cleaned data/Transaction_data_cleaned.csv")
users_df_cleaned=pd.read_csv("cleaned data/users_data_cleaned.csv")

df = trans_df_cleaned.merge(card_df_cleaned, on=['card_id', 'client_id'], how='left')
df = df.merge(users_df_cleaned, left_on='client_id', right_on='id', how='left')
df['is_fraud'] = df['errors'].apply(lambda x: 0 if str(x).strip().lower() == 'none' else 1)  # Label fraud
print('fraud!!!!!!!!!!!!!')
fraud_df = df[df['is_fraud'] == 1]
print(fraud_df.head(5))
fraud_df.to_csv('fraud.csv')
print('doneeee')
print(df['amount'])
df['amount'] = df['amount'].astype(str).str.strip()
df['amount'] = df['amount'].str.replace(r'[^\d.]', '', regex=True)
df = df[df['amount'].str.match(r'^\d+(\.\d+)?$', na=False)]
df['amount'] = df['amount'].astype(float)

df['total_debt'] = df['total_debt'].astype(str).str.strip()
df['total_debt'] = df['total_debt'].str.replace(r'[^\d.]', '', regex=True)
df = df[df['total_debt'].str.match(r'^\d+(\.\d+)?$', na=False)]
df['total_debt'] = df['total_debt'].astype(float)


df['per_capita_income'] = df['per_capita_income'].astype(str).str.strip()
df['per_capita_income'] = df['per_capita_income'].str.replace(r'[^\d.]', '', regex=True)
df = df[df['per_capita_income'].str.match(r'^\d+(\.\d+)?$', na=False)]
df['per_capita_income'] = df['per_capita_income'].astype(float)




df['yearly_income'] = df['yearly_income'].astype(str).str.strip()
df['yearly_income'] = df['yearly_income'].str.replace(r'[^\d.]', '', regex=True)
df = df[df['yearly_income'].str.match(r'^\d+(\.\d+)?$', na=False)]
df['yearly_income'] = df['yearly_income'].astype(float)

df['credit_limit'] = df['credit_limit'].astype(str).str.strip()
df['credit_limit'] = df['credit_limit'].str.replace(r'[^\d.]', '', regex=True)
df = df[df['credit_limit'].str.match(r'^\d+(\.\d+)?$', na=False)]
df['credit_limit'] = df['credit_limit'].astype(float)

df['date'] = pd.to_datetime(df['date'])

print("### Cleaned & Merged Data ###")
print(df.head())


print(df['merchant_state'].nunique())
print(df['card_brand'].nunique())
print(df['card_type'].nunique())
print(df['card_on_dark_web'].nunique())
print('errors!!!!!!')
print(df['is_fraud'].nunique())
print(df['is_fraud'].value_counts())
print(df['errors'].value_counts())



for col in df.columns:
    if df[col].dtype == 'object':  
        df[col] = df[col].map(lambda x: str(x).replace('$', '').strip() if pd.notnull(x) else x)


features = [
    'amount',                
    'merchant_state',        
    'card_brand',           
    'card_type',            
    'credit_limit',         
    'credit_score',         
    'num_credit_cards'      
]

top_states = df['merchant_state'].value_counts().nlargest(10).index
df['merchant_state'] = df['merchant_state'].apply(lambda x: x if x in top_states else 'Other')

df_model = df[features + ['is_fraud']]


    
df_model = pd.get_dummies(df_model, columns=[
'merchant_state', 'card_brand', 'card_type'
], drop_first=True)
from imblearn.over_sampling import SMOTE
X = df_model.drop('is_fraud', axis=1)
y = df_model['is_fraud']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model Evaluation:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

import joblib

joblib.dump(model, 'random_forest_fraud_model.pkl')
print("Model saved as 'random_forest_fraud_model.pkl'")
joblib.dump(model, 'xgboost_fraud_model.pkl')
print("Model saved as 'xgboost_fraud_model.pkl'")
