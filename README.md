import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

os.chdir("C:\\Users\\kkeshav\\Downloads")
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
sleep_df = pd.read_csv('wellbeing_and_lifestyle.csv')
dreams_df = pd.read_csv('dreams.csv')
wellbeing_df = pd.read_csv('Sleep_health_and_lifestyle.csv')
sleep_df.head()
sleep_df.head()
dreams_df.head()
sleep_df.fillna(method='ffill', inplace=True)
wellbeing_df.fillna(method='ffill', inplace=True)
dreams_df['dreams_text'] = dreams_df['dreams_text'].fillna('').apply(str.lower)
dreams_df.dropna(subset=['dreams_text'], inplace=True)
# Exploratory Data Analysis (EDA)
print("Sleep Health Dataset Overview:")
print(sleep_df.info())
print(sleep_df.describe())
print("Wellbeing Dataset Overview:")
print(wellbeing_df.info())
print(wellbeing_df.describe())
print("Dreams Dataset Overview:")
print(dreams_df.info())
sleep_df['FRUITS_VEGGIES'] = pd.to_numeric(sleep_df['FRUITS_VEGGIES'], errors='coerce')
sleep_df['DAILY_STRESS'] = pd.to_numeric(sleep_df['DAILY_STRESS'], errors='coerce')
sleep_df['ACHIEVEMENT'] = pd.to_numeric(sleep_df['ACHIEVEMENT'], errors='coerce')
sleep_df['DAILY_STEPS'] = pd.to_numeric(sleep_df['DAILY_STEPS'], errors='coerce')
sleep_df['WORK_LIFE_BALANCE_SCORE'] = pd.to_numeric(sleep_df['WORK_LIFE_BALANCE_SCORE'], errors='coerce')
# Feature Engineering
# Create 'Wellbeing Score' and 'Sleep Quality Score' based on specific columns
sleep_df['Wellbeing_Score'] = (
    sleep_df['FRUITS_VEGGIES'] * 0.1 + 
    sleep_df['DAILY_STRESS'] * -0.2 + 
    sleep_df['ACHIEVEMENT'] * 0.3 + 
    sleep_df['DAILY_STEPS'] * 0.2 + 
    sleep_df['WORK_LIFE_BALANCE_SCORE'] * 0.3
)

wellbeing_df['Sleep_Quality_Score'] = (
    wellbeing_df['Sleep Duration'] * 0.5 +
    wellbeing_df['Quality of Sleep'] * 0.5
)
# Sentiment Analysis on Dreams Data
dreams_df['dream_sentiment'] = dreams_df['dreams_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
sleep_df.columns = sleep_df.columns.str.lower()
wellbeing_df.columns = wellbeing_df.columns.str.lower()
wellbeing_df['age'] = wellbeing_df['age'].astype(str)
# Step 1: Map age ranges in sleep_df to representative ages
age_mapping = {
    'Less than 20': '19',  # or any representative value
    '21 to 35': '28',      # or the midpoint
    '36 to 50': '43',      # or the midpoint
    '51 or more': '55'     # or any representative value
}

sleep_df['age'] = sleep_df['age'].map(age_mapping)

# Step 2: Strip whitespace from gender columns
sleep_df['gender'] = sleep_df['gender'].str.strip()
wellbeing_df['gender'] = wellbeing_df['gender'].str.strip()
# Merge Datasets based on 'Age' and 'Gender' as common features
merged_df = pd.merge(sleep_df, wellbeing_df, on=['age', 'gender'], how='left')
merged_df = pd.merge(merged_df, dreams_df, left_index=True, right_index=True, how='left')
dreams_df.head()
merged_df.columns
# Select relevant columns for modeling
features = merged_df[['sleep_quality_score', 'wellbeing_score', 'stress level', 'dream_sentiment']]
target = merged_df['sleep disorder']  # Assuming 'Sleep Disorder' is the column to predict
features.tail()
print("Features shape:", features.shape)
print("Target shape:", target.shape)
target.isnull().sum()
merged_df = merged_df.dropna(subset=['dream_sentiment'])
merged_df = merged_df.dropna(subset=['stress level'])
merged_df = merged_df.dropna(subset=['sleep disorder'])
# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Prediction and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", report)
# Visualization
# Plot feature importances
feature_importances = model.feature_importances_
features_list = features.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features_list)
plt.title("Feature Importances in RandomForest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
# Function to analyze dreams and provide sentiment score
def analyze_dream(dream_text):
    sentiment = sia.polarity_scores(dream_text)
    sentiment_score = sentiment['compound']
    
    if sentiment_score >= 0.05:
        sentiment_label = 'positive'
    elif sentiment_score <= -0.05:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'
        
    return sentiment_label, sentiment_score
# Function to generate recommendations based on scores
def generate_recommendations(wellbeing_score, sleep_quality_score, dream_sentiment):
    recommendations = []
    
    # Check overall wellbeing and provide general recommendations
    if wellbeing_score < 3:
        recommendations.append("Consider focusing on a balanced diet and increasing physical activity.")
    if sleep_quality_score < 3:
        recommendations.append("Try to improve sleep hygiene, such as limiting screen time before bed.")
    
    # Provide insights based on dream sentiment
    if dream_sentiment == 'negative':
        recommendations.append("You may be experiencing stress or anxiety. Consider mindfulness exercises or talking to a mental health professional.")
    elif dream_sentiment == 'positive':
        recommendations.append("Keep up with positive practices, as it seems to reflect in your dream patterns.")
    
    return recommendations
# Example of using the functions with sample input
sample_dream_text = "I was flying over beautiful mountains and felt very calm."
sentiment_label, sentiment_score = analyze_dream(sample_dream_text)
recommendations = generate_recommendations(
    wellbeing_score=4.2,  # Sample score
    sleep_quality_score=3.5,  # Sample score
    dream_sentiment=sentiment_label
)

print("\nDream Analysis:")
print("Sentiment Label:", sentiment_label)
print("Sentiment Score:", sentiment_score)

print("\nRecommendations:")
for rec in recommendations:
    print("-",rec)
