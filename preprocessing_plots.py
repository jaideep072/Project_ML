import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use('Agg')
sns.set(style="whitegrid")

DATA_PATH = r"C:\Users\jdeep\OneDrive\Desktop\Project_ML-master\placement_predict_50k Dataset (3)(in) (1).csv"
PLOT_DIR = os.path.join(r"C:\Users\jdeep\OneDrive\Desktop\Project_ML-master", "static", "preprocessing_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def generate_plots():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["PlacementStatus"])

    plot_counter = 1

    def save_plot(fig, title):
        nonlocal plot_counter
        fig.tight_layout()
        filepath = os.path.join(PLOT_DIR, f"plot_{plot_counter}.png")
        fig.savefig(filepath)
        plot_counter += 1
        plt.close(fig)
        return f"preprocessing_plots/plot_{plot_counter - 1}.png"

    # --- 1. Outlier Fix (MinMaxScaler on CodingTestScore) ---
    feature1 = ["CodingTestScore"]
    scaler1 = MinMaxScaler()
    
    # We will plot the before and after for the training data
    before_data1 = train_df[feature1].copy()
    after_data1 = scaler1.fit_transform(before_data1)
    
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    sns.histplot(before_data1[feature1[0]], kde=True, ax=axes1[0], color="blue")
    axes1[0].set_title(f"Before MinMax Scaling: {feature1[0]}")
    sns.histplot(after_data1[:, 0], kde=True, ax=axes1[1], color="green")
    axes1[1].set_title(f"After MinMax Scaling: {feature1[0]}")
    save_plot(fig1, "Outlier Fix (MinMax on CodingTestScore)")

    # --- 2. Standard Scaling ---
    feature2 = ["CGPA", "SoftSkillsRating", "AttendancePercent", "AptitudeTestScore"]
    scaler2 = StandardScaler()
    
    before_data2 = train_df[feature2].copy()
    after_data2 = pd.DataFrame(scaler2.fit_transform(before_data2), columns=feature2)
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    axes2 = axes2.flatten()
    for i, col in enumerate(feature2):
        sns.kdeplot(before_data2[col], ax=axes2[i], color="blue", label="Before")
        # For standard scaling, the x-axis scale changes completely, so a dual-axis or side-by-side is better
        # But we'll just plot them on the same axis with twinx or separate plots. 
        # Separate plots is clearer.
    plt.close(fig2)

    # Let's do side-by-side for each feature instead
    for col in feature2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
        sns.histplot(before_data2[col], kde=True, ax=axes[0], color="blue")
        axes[0].set_title(f"Before Standard Scaling: {col}")
        sns.histplot(after_data2[col], kde=True, ax=axes[1], color="orange")
        axes[1].set_title(f"After Standard Scaling: {col}")
        save_plot(fig, f"Standard Scaling: {col}")

    # --- 3. Minimax Scaling ---
    feature3 = ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore", "Internships"]
    scaler3 = MinMaxScaler()
    
    before_data3 = train_df[feature3].copy()
    after_data3 = pd.DataFrame(scaler3.fit_transform(before_data3), columns=feature3)
    
    for col in feature3:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
        sns.histplot(before_data3[col], kde=True, ax=axes[0], color="blue")
        axes[0].set_title(f"Before MinMax Scaling: {col}")
        sns.histplot(after_data3[col], kde=True, ax=axes[1], color="green")
        axes[1].set_title(f"After MinMax Scaling: {col}")
        save_plot(fig, f"MinMax Scaling: {col}")

    print("All plots generated successfully.")

if __name__ == "__main__":
    generate_plots()
