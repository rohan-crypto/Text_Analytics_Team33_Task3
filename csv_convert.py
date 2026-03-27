import pandas as pd
df = pd.read_pickle('data_preprocessed.pkl')
df.to_csv('data_preprocessed_csv.csv', index=False)

