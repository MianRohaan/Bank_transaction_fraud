import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

card_df = pd.read_csv("E:/OneDrive - Systems Limited/Desktop/Bank transaction fraud/cards_data.csv")
trans_df = pd.read_csv("E:/OneDrive - Systems Limited/Desktop/Bank transaction fraud/transactions_data.csv")
users_df = pd.read_csv("E:/OneDrive - Systems Limited/Desktop/Bank transaction fraud/users_data.csv")
card_df.columns = card_df.columns.str.strip()
trans_df.columns = trans_df.columns.str.strip()
users_df.columns = users_df.columns.str.strip()

print(trans_df.isnull().sum())
print(trans_df['errors'].value_counts())

trans_df['merchant_state'] = trans_df['merchant_state'].fillna('Unknown')
trans_df['zip'] = trans_df['zip'].fillna(0)
trans_df['errors'] = trans_df['errors'].fillna('none')

print(trans_df['errors'].value_counts())

card_df_cleaned = card_df.drop_duplicates()
trans_df_cleaned = trans_df.drop_duplicates()
users_df_cleaned = users_df.drop_duplicates()

if 'card_id' not in card_df_cleaned.columns and 'id' in card_df_cleaned.columns:
    card_df_cleaned.rename(columns={'id': 'card_id'}, inplace=True)
os.makedirs("cleaned data", exist_ok=True)
card_df_cleaned.to_csv("cleaned data/cards_data_cleaned.csv", index=False)
trans_df_cleaned.to_csv("cleaned data/Transaction_data_cleaned.csv", index=False)
users_df_cleaned.to_csv("cleaned data/users_data_cleaned.csv", index=False)

print('hello world')