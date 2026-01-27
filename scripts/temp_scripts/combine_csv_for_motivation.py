#!/usr/bin/env python3
"""
Combine a new CSV file with a parent CSV file.
Edit the paths below before running.
"""

import pandas as pd
from pathlib import Path

# ============ EDIT THESE PATHS ============
PARENT_CSV = "logs/motivation_baseline_mbpp_arrival_rate_100.csv"
NEW_CSV = "logs/motivation_baseline_mbpp_arrival_rate_150.csv"
# ==========================================

def main():
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    parent_path = project_root / PARENT_CSV
    new_path = project_root / NEW_CSV

    print(f"Parent CSV: {parent_path}")
    print(f"New CSV: {new_path}")

    # Load CSVs
    parent_df = pd.read_csv(parent_path)
    new_df = pd.read_csv(new_path)

    print(f"\nParent rows: {len(parent_df)}")
    print(f"New rows: {len(new_df)}")

    # Combine
    combined_df = pd.concat([parent_df, new_df], ignore_index=True)
    print(f"Combined rows: {len(combined_df)}")

    # Save back to parent
    combined_df.to_csv(parent_path, index=False)
    print(f"\nSaved combined CSV to: {parent_path}")

    # Show arrival rates in combined file
    if 'arrival_rate' in combined_df.columns:
        rates = sorted(combined_df['arrival_rate'].unique())
        print(f"Arrival rates in combined file: {rates}")

if __name__ == "__main__":
    main()
