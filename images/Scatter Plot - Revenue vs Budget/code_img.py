import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import requests
from io import StringIO # StringIO is used for text-based in-memory "files"

csv_url = "https://mailuc-my.sharepoint.com/:x:/g/personal/sharmarz_mail_uc_edu/Ef5POYmr8M1PoXaiJbh-QlEBLkKKf4K8AHwhm--qsSnixA?download=1"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response = requests.get(csv_url, headers=headers, timeout=90)
response.raise_for_status()
try:
    csv_text_content = response.content.decode('utf-8')
    print("Successfully decoded content as UTF-8.")
except UnicodeDecodeError:
    print("UTF-8 decoding failed. Trying 'latin1' (iso-8859-1)...")
    try:
        csv_text_content = response.content.decode('latin1')
        print("Successfully decoded content as 'latin1'.")
    except Exception as decode_err:
        print(f"Could not decode file content. Please check the file's encoding. Error: {decode_err}")
        raise # Re-raise the exception if decoding fails

movies = pd.read_csv(StringIO(csv_text_content), dtype={10: str})
movies['revenue'] = pd.to_numeric(movies['revenue'], errors='coerce')
movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce')
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')

cleaned_movies = movies.dropna(subset=['revenue', 'budget', 'release_date'])

cleaned_movies = cleaned_movies[(cleaned_movies['revenue'] > 0) & (cleaned_movies['budget'] > 0)]

df = movies
# Convert budget and revenue to numeric, forcing errors to NaN
df["budget"] = pd.to_numeric(df["budget"], errors='coerce')
df["revenue"] = pd.to_numeric(df["revenue"], errors='coerce')
df_numeric = df[["budget", "revenue", "runtime", "vote_average", "vote_count"]].dropna()

# Scatter plot: Revenue vs. Budget
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_numeric["budget"], y=df_numeric["revenue"], alpha=0.5)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Budget (USD)")
plt.ylabel("Revenue (USD)")
plt.title("Revenue vs. Budget")
# Instead of using slope, define two points for the line in log scale
x_vals = [df_numeric['budget'].min(), df_numeric['budget'].max()]
y_vals = [df_numeric['budget'].min(), df_numeric['budget'].max()]  # For a 45-degree line in log-log
plt.plot(x_vals, y_vals, color='red', linestyle='dashed')
plt.show()

import json
# Define valid genre names
valid_genres = {'Thriller', 'Music', 'Romance', 'Science Fiction', 'Action', 'TV Movie', 'Fantasy',
                'History', 'Foreign', 'Documentary', 'Horror', 'Animation', 'Comedy', 'Adventure',
                'Family', 'Mystery', 'Drama', 'War', 'Western', 'Crime'}

def extract_genres(genre_str):
    try:
        genres = json.loads(genre_str.replace("'", "\"")) if isinstance(genre_str, str) else []
        return [genre["name"] for genre in genres if isinstance(genre, dict) and "name" in genre and genre["name"] in valid_genres]
    except (json.JSONDecodeError, ValueError):
        return []  # Return empty list for invalid data

df["all_genres"] = df["genres"].dropna().apply(extract_genres)

# Number of Movies Per Genre
genre_counts = {}
for genre_list in df["all_genres"].dropna():
    for genre in genre_list:
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
# Scatter plot: Revenue vs. Budget
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
selected_genres = ["Action", "Comedy", "Drama", "Science Fiction"]

for i, genre in enumerate(selected_genres):
    df_genre = df[df["all_genres"].apply(lambda x: genre in x)]
    ax = axes[i // 2, i % 2]
    scatter = sns.scatterplot(x=df_genre["budget"], y=df_genre["revenue"], alpha=0.5, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Revenue")
    ax.set_title(f"Revenue vs. Budget ({genre})")

    # Instead of using slope, define two points for the line in log scale
    x_vals = [df_genre['budget'].min(), df_genre['budget'].max()]
    y_vals = [df_genre['budget'].min(), df_genre['budget'].max()]  # For a 45-degree line in log-log
    ax.plot(x_vals, y_vals, color='red', linestyle='dashed')

plt.tight_layout()
plt.show()