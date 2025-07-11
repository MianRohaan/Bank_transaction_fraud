import joblib
import pandas as pd

model = joblib.load('xgboost_fraud_model.pkl')
import pandas as pd

sample_data = pd.DataFrame({
   'amount': [104.10, 28.84, 38.58, -72.00, 37.54],
    'merchant_state': ['FL', 'Unknown', 'Unknown', 'AZ', 'IN'],
    'card_brand': ['Unknown'] * 5,  
    'card_type': ['Unknown'] * 5,   
    'credit_limit': [None] * 5,    
    'credit_score': [850, 743, 743, 761, 822],
    'num_credit_cards': [1, 5, 5, 4, 6]
})

print(sample_data)


encoded_sample = pd.get_dummies(sample_data)
model_features = model.feature_names_in_  
for col in model_features:
    if col not in encoded_sample.columns:
        encoded_sample[col] = 0

encoded_sample = encoded_sample[model_features]
predictions = model.predict(encoded_sample)

for i, pred in enumerate(predictions):
    label = "FRAUD" if pred == 1 else "LEGIT"
    print(f"Sample {i+1} is predicted as: {label}")
