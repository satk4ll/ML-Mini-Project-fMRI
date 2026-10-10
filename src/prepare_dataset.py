
import pandas as pd
from pathlib import Path

# Project paths
DATA_PATH = Path("data/student-por.csv")
OUTPUT_PATH = Path("data/student-por-labelled.csv")

def main():
    # Load the semicolon-separated dataset
    df = pd.read_csv(DATA_PATH, sep=";")

    # Validate required columns
    required = {"Dalc", "Walc"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain Dalc and Walc columns.")

    # Create the binary classification target
    df["high_consumption"] = (
        (df["Dalc"] >= 4) | (df["Walc"] >= 4)
    ).astype(int)

    # Display class distribution
    print("Dataset shape:", df.shape)
    print("\nTarget distribution:")
    print(df["high_consumption"].value_counts().sort_index())

    print("\nTarget proportions:")
    print(
        df["high_consumption"]
        .value_counts(normalize=True)
        .sort_index()
        .round(3)
    )

    # Save the labelled dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved labelled dataset to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

