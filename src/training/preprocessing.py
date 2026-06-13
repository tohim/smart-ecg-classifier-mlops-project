
# PREPROCESSING SCRIPT

# Loading the ECG-Heartbeat Dataset (MIT-BIH, csv files downloaded from Kaggle) and converting it to Numpy-Arrays to quickly load later.
# The dataset contains individual "heartbeats" extracted from longer ECG (electrocardiogram) recordings.
# The data is already resamples/padded to a length of 187 data points and normalized between 0 and 1.
# The last column of each row is the "label": an integer from 0-4 telling us which TYPE of heartbeat this is.
# Normal, or one of 4 types of arrhythmia / abnormal heartbeat.

import pandas as pd
import numpy as np

# Names for the 5 classes, position in list corresponds to integer label
CLASS_NAMES = [
    "Normal (N)",                               # label 0
    "Supraventricular Premature (S)",           # label 1
    "Premature Ventricular Contraction (V)",    # label 2
    "Fusion of Ventricular and Normal (F)",     # label 3
    "Unclassifiable Beat (Q)",                  # label 4
]

def main():

    # Load raw csv files into pandas DataFrames (no header row, so header=None)
    train_df = pd.read_csv("data/raw/mitbih_train.csv", header=None)
    test_df = pd.read_csv("data/raw/mitbih_test.csv", header=None)

    # Split each DataFrame into features (X) and labels (y)
    # Last column contains the class labels, the rest are the features/ signal values
    X_train = train_df.iloc[:, :-1].values.astype("float32")    # iloc[] selects rows/columns by position
    y_train = train_df.iloc[:, -1].values.astype("int64")       # ":"" means all rows, "-1" means only last column, ":-1" means all columns except the last one
    X_test = test_df.iloc[:, :-1].values.astype("float32")      # .values converts pandas object (df) inot a plain numpy array, bc skikit-learn and pytorch expect np arrays
    y_test = test_df.iloc[:, -1].values.astype("int64")         # .astype("float32") converts to 32-bit floating point numbers, common format for ML models
                                                                # .astype("int64") converts to 64-bit integers, common format for class labels, pytorch expects 
    
    # Quick "did loading work correctly?" check
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Print the class distribution (examples per class) 
    print("\nClass distribution in training set:")

    # Loop counts how many training examples belong to each class - shows whether the dataset is balanced or imbalanced 
    # the loop works by iterating over each class index (i) and name, then counting how many examples in y_train have that class label (i) (using sum of boolean array)
    # and calculating the percentage of the total training set that represents.
    # important to know as it heavily affects evaluation metrics and model performance

    for i, name in enumerate(CLASS_NAMES):       # enumerate(CLASS_NAMES) --> yields pairs (index, name) e.g. (0, "Normal (N)"), (1, "Supraventricular Premature (S)"), etc.
        count = (y_train == i).sum()             # y_train == i --> creates boolean array same length as y_train. True where label equals i, false elsewhere.
                                                 # .sum() counts how many True values there are, which is the same as counting how many examples belong to class i.
        pct = 100 * count / len(y_train)         
        print(f" {i} - {name}: {count:6d} ({pct:5.1f}%)")   # f-strings embed variables directly in the string. {count:6d} means print count as an integer, at 
                                                            # least 6 characters wide (for alignment). 
                                                            # {pct:5.1f} means print pct as a float with 1 decimal place, 
                                                            # at least 5 characters wide.

    # Save the processed arrays to disk (numpy binary format)
    # np.save() writes a numpy array to a ".npy" file
    # Advantages over re-reading the CSV every time:
        # 1. Much faster load (no text parsing)
        # 2. Preserves data types and shapes exactly (float32, int64, etc.)
    
    # We save 4 files: X_train, y_train, X_test, y_test, in the "processed" subfolder.
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_test.npy", y_test)

    print("\nFinished preprocessing and saved data in data/processed/")


if __name__ == "__main__":
    main()
