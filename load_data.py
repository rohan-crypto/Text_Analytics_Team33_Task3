# load_data.py
from datasets import load_dataset
import pandas as pd
import re

def load_arxiv_data():
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train") # "train" here means entire dataset (it is just named that way)
    df = ds.to_pandas()
    
    # Filter for AI/ML/NLP
    target_cats = {"cs.AI", "cs.LG", "cs.CL"} # we only want AI/ML/NLP papers, so we filter by category
    df = df[df["categories"].apply(lambda cats: bool(set(cats) & target_cats))].copy()
    
    # arXiv IDs encode the submission date e.g. 0704.0001 = April 2007, 1512.03385 = December 2015. 
    # The format is always YYMM.XXXXX 
    # Extract year
    def extract_year(arxiv_id):
        match = re.match(r"(\d{2})(\d{2})\.", str(arxiv_id)) # regex pattern that captures YYMM.
        if match:
            yy = int(match.group(1)) # gets the first captured group (YY)
            return 2000 + yy if yy < 90 else 1900 + yy # converts years such as, 07 → 2007, 21 → 2021 and 89 → 1989
        return None
    
    df["year"] = df["id"].apply(extract_year)
    df = df[df["year"].notna()].copy() # drops any rows where year extraction failed (returned None)
    df["year"] = df["year"].astype(int) # converts year from float to integer
    
    return df

if __name__ == "__main__":
    print("Loading data...")
    df = load_arxiv_data()
    print(f"Shape: {df.shape}")
    print(f"Years: {df['year'].min()} to {df['year'].max()}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head(3))
    
    