import pandas as pd
import csv

input_file = '/home/aarohisd1974/Documents/IITJ_4th_year/NLU/bbc-news-data.csv'
output_file = 'sports_vs_politics.csv'

try:
    # sep='\t' tells pandas to split by Tabs instead of Commas
    df = pd.read_csv(input_file, sep='\t', engine='python')
    
    print("--- Debug Info ---")
    print(f"Columns found: {df.columns.tolist()}")
    
    # Standardize category names to lowercase
    target_categories = ['sport', 'politics']
    
    # Filter (BBC dataset uses 'category' as the header)
    mask = df['category'].str.lower().isin(target_categories)
    filtered_df = df[mask]

    # Save to a clean CSV (using actual commas this time for your ML models)
    filtered_df.to_csv(output_file, index=False)
    
    print("\n--- Success ---")
    print(f"Filtered dataset saved to: {output_file}")
    print(f"Total rows kept: {len(filtered_df)}")
    print("Class Distribution:")
    print(filtered_df['category'].value_counts())

except Exception as e:
    print(f"Error: {e}")