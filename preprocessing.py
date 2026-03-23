import re # for regular expressions
import pickle # for saving and loading preprocessed data
import nltk # for natural language processing tasks
import spacy # for advanced NLP tasks like lemmatization
import pandas as pd # for data manipulation
from tqdm.auto import tqdm # for progress bars
from load_data import load_arxiv_data # for loading the arXiv dataset
from nltk.corpus import stopwords # for removing common words that don't add much meaning

nltk.download("stopwords") # download the list of stopwords

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tok2vec"]) # load the small English model in spaCy and disable unnecessary components for faster processing
STOPWORDS = set(stopwords.words("english")) # create a set of stopwords for faster lookup

# Function to check the quality of the data, before cleaning.
# This will help us understand the issues in the data and how to address them in the preprocessing step.
def check_data_quality(df):
    print("Data quality check:")

    # We will be first checking the shape
    print(f"Shape: {df.shape}") # prints the number of rows and columns in the DataFrame
    print(f"Total Papers: {len(df)}") # prints the total number of papers (rows) in the DataFrame
    print(f"Total Columns: {len(df.columns)}") # prints the total number of columns in the DataFrame

    # Now we will check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum()) # prints the number of missing values in each column

    # Now we will check for duplicate IDs
    num_duplicate_ids = df["id"].duplicated().sum() # counts the number of duplicate IDs in the "id" column
    print(f"\nDuplicate IDs: {num_duplicate_ids}") # prints the number of duplicate IDs

    # Now we will check for the duplicate abstracts
    num_duplicate_abstracts = df["abstract"].duplicated().sum() # counts the number of duplicate abstracts in the "abstract" column
    print(f"\nDuplicate Abstracts: {num_duplicate_abstracts}") # prints the number of duplicate abstracts

    # There can be empty abstracts, so we will check for that as well
    num_empty_abstracts = (df["abstract"].str.strip() == "").sum() # counts the number of abstracts that are empty or contain only whitespace
    print(f"\nEmpty Abstracts: {num_empty_abstracts}") # prints the number of empty abstracts

    # There can also be small abstracts, which may not be useful for training, so we will check for abstracts with less than 10 words
    num_short_abstracts = (df["abstract"].str.split().apply(len) < 10).sum() # counts the number of abstracts that have less than 10 words
    print(f"\nShort Abstracts (<10 words): {num_short_abstracts}") # prints the number of short abstracts

    # Finally, we will check the distribution of papers across years
    print(f"\nYear Range: {df['year'].min()} to {df['year'].max()}") # prints the range of years in the dataset
    print(f"Papers per Year:\n{df['year'].value_counts().sort_index()}") # prints the number of papers published each year, sorted by year

# Function to clean the abstracts in the DataFrame, like removing duplicates, nulls and bad rows.
def clean_dataframe(df):

    original_len = len(df) # store the original number of papers for later comparison
    print("\nCleaning data...")

    # First we will drop any rows with missing abstracts, since those won't be useful for training
    df = df.dropna(subset=["abstract", "id", "year"]).copy() # drops rows where any of the specified columns have missing values and creates a copy of the DataFrame to avoid SettingWithCopyWarning  
    print(f"After dropping missing abstracts: {len(df)} papers. So removed {original_len - len(df)}")

    # Next we will drop duplicate IDs, since those won't add much value to training and can lead to overfitting
    before = len(df) # store the number of papers before dropping duplicates for later comparison
    df = df.drop_duplicates(subset=["id"]).copy() # drops duplicate rows based on the "id" column
    print(f"After dropping duplicate IDs: {len(df)} papers. So removed {before - len(df)}")

    # Now we will drop duplicate abstracts
    before = len(df) # store the number of papers before dropping duplicates for later comparison
    df = df.drop_duplicates(subset=["abstract"]).copy() # drops duplicate rows based on the "abstract" column
    print(f"After dropping duplicate abstracts: {len(df)} papers. So removed {before - len(df)}")

    # Dropping empty abstracts
    before = len(df) # store the number of papers before dropping empty abstracts for later comparison
    df = df[df["abstract"].str.strip() != ""].copy() # drops rows where the "abstract" column is empty or contains only whitespace
    print(f"After dropping empty abstracts: {len(df)} papers. So removed {before - len(df)}")

    # Now we will be dropping short abstracts
    before = len(df) # store the number of papers before dropping short abstracts for later comparison
    df = df[df["abstract"].str.split().apply(len) >= 10].copy() # drops rows where the "abstract" column has less than 10 words
    print(f"After dropping short abstracts: {len(df)} papers. So removed {before - len(df)}")

    # Lastly, we will drop papers from invalid years, like before 1990 and after 2025
    before = len(df) # store the number of papers before dropping invalid years for later comparison
    df = df[df["year"].between(1990, 2025)].copy() # keeps only rows where the "year" column is between 1990 and 2025 (inclusive)
    print(f"After dropping invalid years: {len(df)} papers. So removed {before - len(df)}")

    print("\nSummary of cleaning:")
    print(f"Total papers removed: {original_len - len(df)}") # prints the total number of papers removed during cleaning
    print(f"Final number of papers: {len(df)}") # prints the final number of papers after cleaning

    return df.reset_index(drop=True) # resets the index of the DataFrame after cleaning and drops the old index

# We will divide the preprocessing into two steps: basic cleaning and advanced cleaning.
# Then those functions will be called in a main function that will run the entire preprocessing pipeline.
# This function will lowercase the abstracts and remove noise.
def basic_cleaning(text):
    text = text.lower() # convert text to lowercase
    text = re.sub(r"\$.*?\$", " ", text) # remove LaTeX math expressions (anything between $...$)
    text = re.sub(r"http\S+|www\S+", " ", text) # remove URLs (anything starting with http://, https://, or www.)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text) # remove non-alphabetic characters (keep only letters, numbers and hyphens)
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces and trim leading/trailing whitespace

    return text

# This function will lemmatize and remove stopwords from the abstracts.
def advanced_cleaning(text):
    doc = nlp(text) # process the text with spaCy to create a Doc object
    tokens = [token.text.lower() for token in doc if token.text.lower() not in STOPWORDS and len(token.text) > 2 and not token.is_space] # lemmatize tokens and filter out stopwords, short tokens (<=2 characters), and whitespace tokens
    return " ".join(tokens) # join the cleaned tokens back into a single string

# This function will run the entire preprocessing pipeline on the abstracts in the DataFrame.
def preprocess_data(text):
    text = basic_cleaning(text) # apply basic cleaning to the text
    text = advanced_cleaning(text) # apply advanced cleaning to the text
    return text

if __name__ == "__main__":
    # Load the data
    print("Loading data...")
    df = load_arxiv_data() # load the arXiv dataset into a DataFrame

    # Checking the quality of the data before cleaning
    check_data_quality(df) # perform a data quality check on the loaded DataFrame

    # Now cleaning the dataframe
    df = clean_dataframe(df) # clean the DataFrame by removing duplicates, nulls,

    # Now checking quality after cleaning
    print("\nAfter cleaning:")
    print(f"Shape: {df.shape}") # prints the number of rows and columns in the cleaned DataFrame
    print(f"Nulls remaining:\n{df.isnull().sum()}") # prints the number of missing values in each column after cleaning
    print(f"Duplicates remaining: {df.duplicated(subset=['id', 'abstract']).sum()}") # counts the number of duplicate rows based on the "id" and "abstract" columns after cleaning

    # Preprocessing the abstracts
    print("\nPreprocessing abstracts...")
    
    # We will be doing in batches using Spacy's nlp.pipe for faster processing and tqdm for progress bars
    texts = df["abstract"].tolist() # convert the "abstract" column to a list of strings for processing
    cleaned = []

    texts_cleaned = [basic_cleaning(text) for text in texts] # first we will apply basic cleaning to all abstracts in a list comprehension for faster processing

    # Then we apply spaCy for batch processing for lemmatization and stopword removal
    for doc in tqdm(nlp.pipe(texts_cleaned, batch_size=64), total=len(texts_cleaned)):
        tokens = [
                    token.lemma_.lower()
                    for token in doc
                    if token.text.lower() not in STOPWORDS
                    and len(token.text) > 2
                    and not token.is_space
                ]
        cleaned.append(" ".join(tokens)) # join the cleaned tokens back into a single string and add to the cleaned list
    
    df["cleaned_abstract"] = cleaned # add the cleaned abstracts as a new column in the DataFrame

    # Assign time periods
    # Using even 5-year windows to capture key shifts in AI/ML/NLP history
    def assign_period(year):
        if year < 2000:
            return "pre-2000"
        elif year < 2005:
            return "2000-2004"
        elif year < 2010:
            return "2005-2009"
        elif year < 2015:
            return "2010-2014"
        elif year < 2020:
            return "2015-2019"
        else:
            return "2020-2025"
    
    df["period"] = df["year"].apply(assign_period) # create a new column "period" by applying the assign_period function to the "year" column
    print(f"\nPapers per period:\n{df['period'].value_counts().sort_index()}") # prints the number of papers in each time period, sorted by period

    # Just a sample check to see how the cleaned abstracts look like
    print("\nSample before:")
    print(df["abstract"].iloc[0][:200]) # print the first 200 characters of the original abstract of the first paper

    print("\nSample after:")
    print(df["cleaned_abstract"].iloc[0][:200]) # print the first 200 characters of the cleaned abstract of the first paper

    # Saving and creating a pickle file of the cleaned DataFrame for later use in training
    print("\nSaving pickle...")
    df.to_pickle("data_preprocessed.pkl")
    print("\nSaved the pickle file as data_preprocessed.pkl")
    print(f"Final shape of the cleaned DataFrame: {df.shape}") # prints the shape of the cleaned DataFrame
    print(f"Columns in the cleaned DataFrame: {df.columns.tolist()}") # prints the list of columns in the cleaned DataFrame 

