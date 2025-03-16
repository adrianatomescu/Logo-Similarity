import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

folder_path = os.path.join(script_dir, "Logo_Similarity")

input_file = os.path.join(folder_path, "logos.snappy.parquet")
output_file = os.path.join(folder_path, "logos.csv")

df = pd.read_parquet(input_file)
df.to_csv(output_file, index=False)

print("Conversie reusita.")
