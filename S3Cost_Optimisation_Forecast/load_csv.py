# 📊 Load your sample AWS S3 cost dataset
import pandas as pd
import os

# 👇 Replace with your CSV file path (or place CSV in same directory)
csv_file = r"D:\Python\S3Analysis\myenv\aws_s3_spend_500_rows.csv"

# ✅ Check if the file exists
if not os.path.exists(csv_file):
    print(f"❌ File not found: {csv_file}")
    exit()

# 📥 Load the CSV file
try:
    df = pd.read_csv(csv_file)
    
    print("✅ CSV loaded successfully!")
except Exception as e:
    print("❌ Error loading CSV:", e)
    exit()

# 📊 Show basic info
print("\n📌 First 5 rows:")
print(df.head())

print("\n📌 Columns:", df.columns.tolist())
print(f"📈 Total Rows: {len(df)}")

def load_data():
    df = pd.read_csv("aws_s3_spend_500_rows.csv")
    df.columns = [col.strip().capitalize() for col in df.columns]  # Capitalize for consistency
    print("📊 CSV loaded successfully!")
    return df