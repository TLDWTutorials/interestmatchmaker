# Junior PM & Senior PM Matching System

## 📌 Overview

This project implements a **matching system** that pairs **Junior Project Managers (PMs)** with **Senior PMs** based on **cosine similarity** of their interests, skillsets, age, gender, and tenure at the company.

The script processes employee profiles from a CSV file, encodes their features, calculates similarity scores, and generates optimal **top 3** matches for each Junior PM.

## 🚀 Features

- **Data Preprocessing:** Handles categorical variables (`Interests`, `Skillsets`, `Race`, `Gender`) and normalizes numerical variables (`Age`, `Years at Office`).
- **Cosine Similarity Matching:** Finds employees with similar interests and skillsets.
- **Adjusted Similarity Scoring:** Incorporates **age proximity**, **years at office**, and **gender similarity** to refine matches.
- **Automated Matching System:** Finds **top 3 Senior PMs** for each Junior PM.
- **Output Generation:** Writes the results to a text file (`optimal_matches_junior_pm_top_3_senior_pm.txt`).

## 📂 Dataset Format

The input dataset (`mock_profiles_100.csv`) should have the following structure:

| Name  | Age | Gender | Race | Years at Office | Interests | Skillsets | PM_Type  |
|-------|-----|--------|------|----------------|-----------|-----------|----------|
| Alice | 30  | Female | White | 3              | `['AI', 'Data Science']` | `['Python', 'Machine Learning']` | Junior PM |
| Bob   | 45  | Male   | Asian | 10             | `['Finance', 'Risk Analysis']` | `['SQL', 'Excel']` | Senior PM |

📌 **Note:** The `Interests` and `Skillsets` fields should be **lists** (not strings), formatted as Python lists within the CSV file.

## 🛠️ Installation & Setup

1. Install required dependencies
```

pip install pandas scikit-learn numpy
```
2. Load and Process the Data

- Reads mock_profiles_100.csv.
- Converts Interests and Skillsets from string representation to Python lists.
- Applies one-hot encoding for Interests and Skillsets.
- Normalizes Age and Years at Office.
- Encodes categorical variables (e.g., Race, Gender).

3. Compute Similarity Matrix

- Uses cosine similarity to compare employee profiles.
- Adjusts similarity scores using:
- Years at Office Similarity: Higher similarity for employees with similar tenure.
- Gender Matching: Preference for same-gender pairing.
- Age Proximity: Higher scores for employees within a 5-year age range.

4. Find Optimal Matches

- Matches each Junior PM with the top 3 most suitable Senior PMs based on adjusted similarity scores.
- Saves the results to a text file.
  
