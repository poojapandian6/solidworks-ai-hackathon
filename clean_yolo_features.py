import pandas as pd
import re

IN_CSV = "yolo_features_train.csv"
OUT_CSV = "yolo_features_train_clean.csv"

df = pd.read_csv(IN_CSV)

def clean_value(x):
    if isinstance(x, str):
        # extract number from tensor(123.45, device='cuda:0')
        m = re.search(r"tensor\(([-0-9.]+)", x)
        if m:
            return float(m.group(1))
    return x

for col in df.columns:
    df[col] = df[col].apply(clean_value)

df.to_csv(OUT_CSV, index=False)

print("✅ Cleaned CSV saved as", OUT_CSV)
print(df.dtypes)
