import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset
df = pd.read_csv('mock_profiles_100.csv')

# Ensure the interests and skillsets columns are lists
df['Interests'] = df['Interests'].apply(eval)
df['Skillsets'] = df['Skillsets'].apply(eval)

# One-hot encode categorical variables for interests and skillsets
mlb_interests = MultiLabelBinarizer()
interests_encoded = mlb_interests.fit_transform(df['Interests'])

mlb_skillsets = MultiLabelBinarizer()
skillsets_encoded = mlb_skillsets.fit_transform(df['Skillsets'])

# Normalize age and years at ccmr
age_normalized = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
years_at_ccmr_normalized = (df['Years at office'] - df['Years at office'].min()) / (df['Years at office'].max() - df['Years at office'].min())

# Concatenate all features into a single feature matrix
features = pd.concat([
    pd.DataFrame(interests_encoded, columns=mlb_interests.classes_),
    pd.DataFrame(skillsets_encoded, columns=mlb_skillsets.classes_),
    age_normalized,
    pd.get_dummies(df[['Race']])
], axis=1)

# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(features)

# Adjust the similarity scores to consider years at company, gender, and age proximity
def adjust_similarity_scores(similarity_matrix, years_at_ccmr_normalized, genders, ages, years_weight=0.1, gender_weight=0.1, age_weight=0.1):
    adjusted_similarity_matrix = np.copy(similarity_matrix)
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            year_diff = abs(years_at_ccmr_normalized[i] - years_at_ccmr_normalized[j])
            year_similarity = 1 - year_diff
            
            gender_similarity = 1 if genders[i] == genders[j] else 0
            age_diff = abs(ages[i] - ages[j])
            age_similarity = 1 if age_diff <= 5 else 0
            
            adjusted_similarity_matrix[i, j] = (
                (1 - years_weight - gender_weight - age_weight) * similarity_matrix[i, j] + 
                years_weight * year_similarity + 
                gender_weight * gender_similarity + 
                age_weight * age_similarity
            )
    return adjusted_similarity_matrix

adjusted_similarity_matrix = adjust_similarity_scores(
    similarity_matrix, 
    years_at_ccmr_normalized, 
    df['Gender'], 
    df['Age']
)

# Calculate a combined suitability score for each employee
df['Suitability_Score'] = (
    0.1 * age_normalized + 
    0.1 * years_at_ccmr_normalized + 
    0.4 * pd.Series([sum(x) for x in interests_encoded]) / len(interests_encoded[0]) +
    0.4 * pd.Series([sum(x) for x in skillsets_encoded]) / len(skillsets_encoded[0])
)

# Function to find the optimal matches for Junior PMs with top 3 Senior PMs
def find_optimal_matches_for_junior_pm(adjusted_similarity_matrix, df, top_n=3):
    matches = {}
    
    junior_pms = df[df['PM_Type'] == 'Junior PM']
    senior_pms = df[df['PM_Type'] == 'Senior PM']
    
    for i, junior_pm in junior_pms.iterrows():
        match_details = []
        for j, senior_pm in senior_pms.iterrows():
            score = adjusted_similarity_matrix[i, j]
            if score > 0:
                match_details.append((senior_pm['Name'], score, senior_pm['Interests'], senior_pm['Skillsets'], senior_pm['Age']))
        
        # Sort match details by similarity score in descending order and get the top_n matches
        match_details.sort(key=lambda x: x[1], reverse=True)
        top_matches = match_details[:top_n]
        
        matches[junior_pm['Name']] = top_matches
    return matches

# Get the optimal matches for Junior PMs
optimal_matches = find_optimal_matches_for_junior_pm(adjusted_similarity_matrix, df)

# Write the results to a text file
output_file = 'optimal_matches_junior_pm_top_3_senior_pm.txt'
with open(output_file, 'w') as file:
    for profile_name, match_list in optimal_matches.items():
        profile_row = df[df['Name'] == profile_name].iloc[0]
        profile_interests = profile_row['Interests']
        profile_skillsets = profile_row['Skillsets']
        profile_age = profile_row['Age']
        file.write(f"Optimal matches for {profile_name} (Interests: {profile_interests}, Skillsets: {profile_skillsets}, Age: {profile_age}):\n")
        for match_name, score, interests, skillsets, age in match_list:
            file.write(f"  {match_name} (Interests: {interests}, Skillsets: {skillsets}, Age: {age}) with similarity score: {score:.4f}\n")
        file.write("\n")

print(f"Optimal matches have been written to {output_file}")

