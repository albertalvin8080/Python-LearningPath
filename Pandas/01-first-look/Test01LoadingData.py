import pandas as pd
import os

os.system("cls")

# df = pd.read_csv("pokemon_data.csv")
# df = pd.read_excel("pokemon_data.xlsx")
df = pd.read_csv("pokemon_data.txt", delimiter="\t")  # TAB
# print(df)

# print(df.head(10)) # First 10 rows
# print("-"*100)
# print(df.tail(10)) # Last 10 rows
# print("-"*100)
# print(df.tail(10)[::-1]) # Descending order / Back to front

print("=" * 100)

# Manipulating Columns
print(df.columns)
print("-" * 100)
print(df["Name"])
print("-" * 100)
print(df["Sp. Atk"][0:5])
print("-" * 100)
print(df[["Name", "Attack", "#"]][0:5])

print("=" * 100)

# Manipulating Rows and Columns
print(df.iloc[160:170])  # 10 rowns
print("-" * 100)
print(df.iloc[2, 1])  # Specific item (R, C)
print(df.iloc[2, 5])  # Specific item (R, C)
