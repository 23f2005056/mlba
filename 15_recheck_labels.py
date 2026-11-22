import pandas as pd

a = pd.read_parquet("out/labels.parquet")
b = pd.read_parquet("out_recheck/labels.parquet")

print("Original shape:", a.shape)
print("Rechecked shape:", b.shape)

print("Same shape:", a.shape == b.shape)
print("Same columns:", list(a.columns) == list(b.columns))
print("Identical content:", a.equals(b))