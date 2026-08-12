import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = r"C:\Users\jdeep\OneDrive\Desktop\Project_ML-master\placement_predict_50k Dataset (3)(in) (1).csv"

df = pd.read_csv(file_path)

numericColumns = ["CGPA", "SoftSkillsRating", "AttendancePercent", "AptitudeTestScore"]
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["PlacementStatus"])
print("\nTraining Data before Standardization")
print(train_df[numericColumns].head())
print("\nTesting Data before Standardization")
print(test_df[numericColumns].head())

scaler = StandardScaler()
train_df[numericColumns] = scaler.fit_transform(train_df[numericColumns])
test_df[numericColumns] = scaler.transform(test_df[numericColumns])
print("\nTraining Data after Standardization")
print(train_df[numericColumns].head())
print("\nTesting Data after Standardization")
print(test_df[numericColumns].head())
