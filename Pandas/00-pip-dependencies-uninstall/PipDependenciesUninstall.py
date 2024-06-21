import pandas as pd
import os

os.system("cls")
# os.system("pip list > dependencies.txt")

df = pd.read_csv("dependencies.txt", sep="\s+", skiprows=[1])
# df = pd.read_csv("dependencies.txt", delimiter="\s+")
# df = pd.read_csv("dependencies.txt", delim_whitespace=True)
# print(df)
print(df.columns)

# print(df["Package"])
filter = df["Package"] != "pip"

filter = (df["Package"] != "pip") & (df["Package"] != "numpy") & (df["Package"] != "pandas")
print(filter)
df_filtered = df[filter]["Package"]
# print(df_filtered)

packages = ' '.join(df_filtered.to_list())
print(f"Packages to uninstall: {packages}")

