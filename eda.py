import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fontTools import annotations
from fontTools.merge import cmap
from pandas.core.dtypes import missing

import matplotlib
matplotlib.use('Agg')

CSV_PATH="placement_predict_50k Dataset (3)(in) (1).csv"
sns.set(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

plot_counter = 1

def show(title=""):
    global plot_counter
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join("static", "eda_plots", f"plot_{plot_counter}.png"))
    plot_counter += 1
    plt.close("all")


#Load data
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f" CSV not found at '{CSV_PATH}'. Updating CSV_PATH at top of scripts ")

data=pd.read_csv(CSV_PATH)
print("="*80)
print("1.DATA LOADED")
print("="*80)
print("Shape:",data.shape)
print("\n First 5 rows:\n",data.head())

#2.Basic info /structure
print("\n"+"="*80)
print("2. BASIC INFO")
print("="*80)
print(data.info())
print("\nColumn dtypes:\n",data.dtypes)
print("\nDescribe (numeric):\n",data.describe())
print("\nDescribe (categorial):\n",data.describe(include="object"))


#3.Missing values
print("\n"+"="*80)
print("#.MISSING VALUES")
print("="*80)
missing=data.isnull().sum()
missing_pct=(missing/len(data))*100
missing_df=pd.DataFrame({"missing_count":missing,"missing_pct": missing_pct})
missing_df=missing_df[missing_df["missing_count"]>0].sort_values(by="missing_count",ascending=False)
print(missing_df)


if not missing_df.empty:
    plt.figure(figsize = (10,5),dpi=100)
    sns.barplot(x=missing_df.index,y=missing_df["missing_pct"])
    plt.xticks(rotation=45,ha="right")
    plt.ylabel("Missing %")
    plt.title("Missing Values by Column")
    show()

#4.DUPLICATES

print("\n"+"="*80)
print("4.DUPLICATE ROWS")
print("="*80)
print("Duplicate rows:",data.duplicated().sum())


#5.TARGET VARIABLE DISTRIBUTION(PlacementStatus)
print("\n"+"="*80)
print("5.TARGET VARIABLE - PlacementStatus")
print("="*80)
print(data["PlacementStatus"].value_counts())

plt.figure(dpi=125)
sns.countplot(x="PlacementStatus",data=data)
plt.xlabel("Placement Status(0 =Not Placed. 1=Placed)")
plt.ylabel("Count")
plt.title("Placement Status Distribution")
show()

#6.Numeric Feature Distributions
print("\n"+"="*80)
print("6.NUMERIC DISTRIBUTIONS")
print("="*80)

hist_cols=["CGPA","AttendancePercent","AptitudeTestScore","SoftSkillsRating","CodingTestScore","MockInterviewScore","Salary Package"]
hist_cols=[c for c in hist_cols if c in data.columns]

data[hist_cols].hist(figsize = (14,10), bins=20)
show("Numeric Feature Distribution")

#Mean line example
plt.figure(dpi=125)
sns.histplot(data["CGPA"] ,kde=True)
plt.axvline(x=np.mean(data["CGPA"]), color="green", linestyle="--", label="CGPA")
plt.legend()
plt.title("CGPA Distribution with Mean")
show()
#7.Outlier Detection (Boxplots)
print("\n"+"="*80)
print("7.OUTLIER DETECTION( BOXPLOTS")
print("="*80)

box_cols=["CGPA","AttendancePercent","AptitudeTestScore","SoftSkillsRating","CodingTestScore","MockInterviewScore","Salary Package"]
box_cols=[c for c in box_cols if c in data.columns]

for col in box_cols:
    plt.figure(figsize = (10,4))
    sns.boxplot(x=data[col],color="skyblue")
    plt.title(f"Boxplot -{col}",fontsize=14)
    show()
# 8. Correlation Heatmap
print("\n"+"="*80)
print("7.Correlation HeatMap")
print("="*80)
corr = data.select_dtypes(include=[np.number]).corr()
print(np.round(corr,2))
plt.figure(figsize=(16, 12) , dpi=100)
sns.heatmap(np.round(corr, decimals=2), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("correlation HeatMap")
show()
#9.Relationship Plots
print("\n" + "=" * 80)
print("9.RELATIONSHIP PLOTS")
print("=" * 80)

# CGPA vs Salary Package
if "CGPA" in data.columns and "Salary Package" in data.columns:
    plt.figure(figsize=(8,6), dpi=100)
    sns.regplot(x="CGPA", y="Salary Package", data=data,
                scatter_kws={"alpha":0.5},
                line_kws={"color":"red"})
    plt.title("CGPA vs Salary Package")
    show()

# Aptitude vs Coding Test Score
if "AptitudeTestScore" in data.columns and "CodingTestScore" in data.columns:
    plt.figure(figsize=(8, 6), dpi=100)
    sns.regplot(x="AptitudeTestScore", y="CodingTestScore", data=data,
                scatter_kws={"alpha": 0.5},
                line_kws={"color": "red"})
    plt.title("Aptitude Test Score vs Coding Test Score")
    show()

#10.Categorical Feature Counts
print("\n" + "=" * 80)
print("10.CATEGORICAL FEATURE COUNTS")
print("=" * 80)

cat_cols = [
    "Gender",
    "City",
    "CollegeTier",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs",
    "CGPA_Tier"
]

cat_cols = [c for c in cat_cols if c in data.columns]

for col in cat_cols:
    print("\n", data[col].value_counts())

    plt.figure(figsize=(8,5), dpi=100)
    sns.countplot(x=col, data=data)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{col} Count")
    show()


#11.Gender vs Placement Status
print("\n" + "=" * 80)
print("11.GENDER VS PLACEMENT STATUS")
print("=" * 80)

if "Gender" in data.columns and "PlacementStatus" in data.columns:
    plt.figure(figsize=(8,5), dpi=100)
    sns.countplot(x="Gender", hue="PlacementStatus", data=data)
    plt.title("Gender vs Placement Status")
    show()


#12.College Tier / Stream vs Placement Status
print("\n" + "=" * 80)
print("12.COLLEGE TIER / STREAM VS PLACEMENT STATUS")
print("=" * 80)

if "CollegeTier" in data.columns and "PlacementStatus" in data.columns:
    plt.figure(figsize=(8,5), dpi=100)
    sns.countplot(x="CollegeTier", hue="PlacementStatus", data=data)
    plt.title("College Tier vs Placement Status")
    show()

if "Stream" in data.columns and "PlacementStatus" in data.columns:
    plt.figure(figsize=(10,6), dpi=100)
    sns.countplot(x="Stream", hue="PlacementStatus", data=data)
    plt.xticks(rotation=45, ha="right")
    plt.title("Stream vs Placement Status")
    show()


#13.SGPA Trend Across Semesters
print("\n" + "=" * 80)
print("13.SGPA TREND ACROSS SEMESTERS")
print("=" * 80)

sgpa_cols = [
    "Sem1_SGPA",
    "Sem2_SGPA",
    "Sem3_SGPA",
    "Sem4_SGPA",
    "Sem5_SGPA",
    "Sem6_SGPA",
    "Sem7_SGPA",
    "Sem8_SGPA"
]

sgpa_cols = [c for c in sgpa_cols if c in data.columns]

if len(sgpa_cols) > 0:
    avg_sgpa = data[sgpa_cols].mean()

    plt.figure(figsize=(10,5), dpi=100)
    plt.plot(avg_sgpa.index, avg_sgpa.values, marker="o")
    plt.xlabel("Semester")
    plt.ylabel("Average SGPA")
    plt.title("Average SGPA Across Semesters")
    plt.grid(True)
    show()


#14.Salary Package Analysis
print("\n" + "=" * 80)
print("14.SALARY PACKAGE ANALYSIS")
print("=" * 80)

if "Salary Package" in data.columns and "PlacementStatus" in data.columns:

    placed = data[data["PlacementStatus"] == 1]

    plt.figure(figsize=(8,5), dpi=100)
    sns.histplot(placed["Salary Package"], kde=True, bins=20)
    plt.title("Salary Distribution for Placed Students")
    show()

if "CollegeTier" in data.columns and "Salary Package" in data.columns:
    plt.figure(figsize=(8,5), dpi=100)
    sns.boxplot(x="CollegeTier", y="Salary Package", data=data)
    plt.title("Salary Package by College Tier")
    show()


#15.Pairplot
print("\n" + "=" * 80)
print("15.PAIRPLOT")
print("=" * 80)

pair_cols = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore",
    "PlacementStatus"
]

pair_cols = [c for c in pair_cols if c in data.columns]

if len(pair_cols) == 5:
    sns.pairplot(data[pair_cols], hue="PlacementStatus")
    show()



