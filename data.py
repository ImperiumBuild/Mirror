import pandas as pd

df1 = pd.read_csv("books_final.csv")
df2 = pd.read_csv("electronics_final.csv")
df3 = pd.read_csv("movies_final.csv")

print(df1.info())
print(df2.info())
print(df3.info())