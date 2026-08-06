import os
import glob
import re
from flask import Flask, render_template
from load_data import load_data, get_data_summary

app = Flask(__name__)


@app.route("/")
def index():
    # Landing page, no section selected yet
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():
    """Loads the dataset (server-side) and renders the summary into the page."""
    error = None
    summary = None

    try:
        # Load the CSV file
        df = load_data()

        # Generate the summary
        summary = get_data_summary(df)

    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error,
    )

@app.route("/eda")
def eda():
    eda_sections = [
        {
            "id": 1,
            "title": "1. Data Loaded",
            "explanation": "The initial step where we load the dataset from the CSV file. We verify the shape (number of rows and columns) and take a quick look at the first few rows to ensure data was read correctly. (No graph generated).",
            "images": []
        },
        {
            "id": 2,
            "title": "2. Basic Info & Structure",
            "explanation": "We check the structural details of the dataset, including data types (numeric vs categorical) and basic summary statistics (mean, min, max, etc.) for all columns. (No graph generated).",
            "images": []
        },
        {
            "id": 3,
            "title": "3. Missing Values",
            "explanation": "Identifying columns that have missing (null) values. In this dataset, there were no missing values, so no imputation was needed and no plot was generated.",
            "images": []
        },
        {
            "id": 4,
            "title": "4. Duplicate Rows",
            "explanation": "Checking for any identical rows in the dataset which might skew our model. The dataset had 0 duplicate rows. (No graph generated).",
            "images": []
        },
        {
            "id": 5,
            "title": "5. Target Variable (Placement Status)",
            "explanation": "This bar chart shows the distribution of our target variable ('PlacementStatus'). It is crucial to check if the dataset is balanced (equal number of placed vs not placed) or imbalanced, as this affects model training.",
            "images": ["eda_plots/plot_1.png"]
        },
        {
            "id": 6,
            "title": "6. Numeric Distributions",
            "explanation": "Histograms for continuous numeric features like CGPA and Test Scores. This helps us understand the spread, central tendency, and skewness of the data. For example, we highlight the mean value for CGPA.",
            "images": ["eda_plots/plot_2.png", "eda_plots/plot_3.png"]
        },
        {
            "id": 7,
            "title": "7. Outlier Detection (Boxplots)",
            "explanation": "Boxplots are used to visually identify outliers in numeric columns. The 'box' represents the interquartile range (middle 50% of data), and the 'whiskers' extend to the rest. Points outside the whiskers are outliers that might negatively impact model performance.",
            "images": [f"eda_plots/plot_{i}.png" for i in range(4, 11)]
        },
        {
            "id": 8,
            "title": "8. Correlation Heatmap",
            "explanation": "A heatmap displaying the Pearson correlation coefficient between all numeric features. Values close to 1 or -1 indicate strong correlation. This helps us find redundant features (multicollinearity) and identify which features correlate most strongly with Placement.",
            "images": ["eda_plots/plot_11.png"]
        },
        {
            "id": 9,
            "title": "9. Relationship Plots",
            "explanation": "Scatter plots with regression lines to explore the direct linear relationship between specific pairs of variables, such as how CGPA correlates with Salary Package.",
            "images": ["eda_plots/plot_12.png", "eda_plots/plot_13.png"]
        },
        {
            "id": 10,
            "title": "10. Categorical Feature Counts",
            "explanation": "Count plots for all categorical variables (Gender, City, Stream, etc.) to visualize the frequency of each category. This shows us the demographic and academic breakdown of the students.",
            "images": [f"eda_plots/plot_{i}.png" for i in range(14, 22)]
        },
        {
            "id": 11,
            "title": "11. Gender vs Placement Status",
            "explanation": "A grouped bar chart comparing the placement rate across genders to see if there is any disparity or trend based on gender.",
            "images": ["eda_plots/plot_22.png"]
        },
        {
            "id": 12,
            "title": "12. College Tier & Stream vs Placement Status",
            "explanation": "These grouped bar charts compare placement success rates based on the student's College Tier (Tier 1, 2, 3) and their Engineering Stream (CS, ECE, Mechanical, etc.).",
            "images": ["eda_plots/plot_23.png", "eda_plots/plot_24.png"]
        },
        {
            "id": 13,
            "title": "13. SGPA Trend Across Semesters",
            "explanation": "A line chart tracking the average SGPA of students across all 8 semesters, revealing how academic performance trends over the 4 years of college.",
            "images": ["eda_plots/plot_25.png"]
        },
        {
            "id": 14,
            "title": "14. Salary Package Analysis",
            "explanation": "Analyzes the distribution of salary packages specifically for students who were placed, and how those packages vary depending on the college tier.",
            "images": ["eda_plots/plot_26.png", "eda_plots/plot_27.png"]
        },
        {
            "id": 15,
            "title": "15. Pairplot",
            "explanation": "A grid of scatter plots showing pairwise relationships between key features simultaneously. The points are colored based on Placement Status, helping us spot clusters and separation boundaries for the classification model.",
            "images": ["eda_plots/plot_28.png"]
        }
    ]
    return render_template("index.html", active="eda", eda_sections=eda_sections)

if __name__ == "__main__":
    app.run(debug=True)