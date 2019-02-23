import pandas as pd

df = pd.read_csv('heart.csv')

df.to_json('data.json')
