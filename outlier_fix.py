import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

file_path = r"C:\Users\jdeep\OneDrive\Desktop\Project_ML-master\placement_predict_50k Dataset (3)(in) (1).csv"

df = pd.read_csv(file_path)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["PlacementStatus"])

scaler = MinMaxScaler()

feature = ["CodingTestScore"]
print("\nTraining Data before scaling")
print(train_df[feature].head())
print("\nTesting Data before scaling")
print(test_df[feature].head())

train_df[feature] = scaler.fit_transform(train_df[feature])
test_df[feature] = scaler.transform(test_df[feature])

print("\nTraining Data after scaling:")
print(train_df[feature].head())
print("\nTesting Data after scaling:")
print(test_df[feature].head())
