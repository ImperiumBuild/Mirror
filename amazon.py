import pandas as pd

df1 = pd.read_csv("data/books_final.csv", low_memory=False)
df2 = pd.read_csv("data/electronics_final.csv", low_memory=False)
df3 = pd.read_csv("data/movies_final.csv", low_memory=False)

df1['category'] = 'books'
df2['category'] = 'electronics'
df3['category'] = 'movies'

df = pd.concat([df1, df2, df3], ignore_index=True)

print(f"Total reviews: {len(df)}")
print(f"Reviews with title: {df['title'].notna().sum()}")
print(f"Unique users: {df['user_id'].nunique()}")

# How many reviews per user?
reviews_per_user = df.groupby('user_id').size()
print(f"Users with 1 review: {(reviews_per_user == 1).sum()}")
print(f"Users with 2 reviews: {(reviews_per_user == 2).sum()}")
print(f"Users with 3+ reviews: {(reviews_per_user >= 3).sum()}")
print(f"Users with 5+ reviews: {(reviews_per_user >= 5).sum()}")