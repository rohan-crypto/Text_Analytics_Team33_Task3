from datasets import load_dataset
import pandas as pd
import re

def load_arxiv_data(sample_n=2000, random_state=42):
    target_cats = {"cs.AI", "cs.LG", "cs.CL", "cs.NE", "stat.ML"}

    def extract_year(arxiv_id):
        match = re.match(r"(\d{2})(\d{2})\.", str(arxiv_id))
        if match:
            yy = int(match.group(1))
            return 2000 + yy if yy < 90 else 1900 + yy
        return None

    # 用 streaming 避免一下子加载整个 200 万数据
    ds = load_dataset(
        "gfissore/arxiv-abstracts-2021",
        split="train",
        streaming=True
    )

    # 打乱，避免只拿到数据集最前面那一段
    ds = ds.shuffle(seed=random_state, buffer_size=10000)

    rows = []
    for row in ds:
        cats = row.get("categories", [])
        if not (set(cats) & target_cats):
            continue

        year = extract_year(row.get("id"))
        if year is None:
            continue

        row["year"] = int(year)
        rows.append(row)

        if len(rows) >= sample_n:
            break

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    print("Loading data...")
    df = load_arxiv_data()
    print(f"Shape: {df.shape}")
    print(f"Years: {df['year'].min()} to {df['year'].max()}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head(3))