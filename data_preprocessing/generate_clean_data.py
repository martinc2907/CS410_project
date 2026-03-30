import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_DATA_DIR = PROJECT_ROOT / "data" / "clean_news_dataset"
DATA_DIR = PROJECT_ROOT / "data" / "news_dataset"

def load_dataset():
    fake_df = pd.read_csv(DATA_DIR / "Fake.csv")
    true_df = pd.read_csv(DATA_DIR / "True.csv")

    true_df['label'] = 'real'
    fake_df['label'] = 'fake'
    df = pd.concat([true_df, fake_df], ignore_index=True)
    return df

def basic_check(df):
    print("Shape of dataset:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    
    # Handle Duplicate values
    df = df.drop_duplicates()
    print("Shape of dataset after removing duplicates:", df.shape)
    # duplicates = df.duplicated().sum()
    # print("\nNumber of duplicate rows:", duplicates)

    # Class distribution
    print("\nClass distribution:\n", df['label'].value_counts())
    return df

def text_length_analysis(df):
    df = df.copy()
    df.loc[:, 'content'] = df['title'] + " " + df['text']
    df.loc[:, 'word_count'] = df['content'].apply(lambda x: len(str(x).split()))

    # Plotting word count distribution
    plt.figure()
    sns.boxplot(x='label', y='word_count', data=df)
    plt.title("Word Count by Class")
    plt.show()

def clean_news_text(text):
    if not isinstance(text, str): return ""

    # Standardize Unicode (quotes, accents) and Remove Reuters boilerplate 
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'^[A-Z]+\s+\(Reuters\)\s+-\s+', '', text)

    # Remove URLs and HTML noise and Normalize whitespace
    text = re.sub(r'http\S+|www\S+|<[^>]*>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_data(df):
  df = df.copy()
  print("Cleaning text... please wait.")
  df['text'] = df['text'].apply(clean_news_text)
  print(f"Size of the data post basic text cleaning: {len(df)}")

  # Remove rows that are empty after cleaning
  df['text'] = df['text'].replace('', np.nan)
  df = df.dropna(subset=['text'])

  print(f"Cleaning complete. Remaining records: {len(df)}")
  return df

def generate_clean_data(df):
    # Preprocess the datasets
    df = basic_check(df)

    # Analyze text length distribution
    text_length_analysis(df)

    # Pre-process the data
    df_clean = preprocess_data(df)

    return df_clean[df_clean['label'] == 'fake'], df_clean[df_clean['label'] == 'true']

if __name__ == "__main__":
    df = load_dataset()
    fake_clean, true_clean = generate_clean_data(df)

    # store the csv in the clean data directory
    fake_clean.to_csv(CLEAN_DATA_DIR / "Fake.csv", index=False)
    true_clean.to_csv(CLEAN_DATA_DIR / "True.csv", index=False)
