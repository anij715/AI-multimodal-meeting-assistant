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
# Bar chart: Original Language Distribution
plt.figure(figsize=(10, 6))
df["original_language"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Original Languages")
plt.xlabel("Language")
plt.ylabel("Count")
plt.show()