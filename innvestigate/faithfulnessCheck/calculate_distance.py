import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

def calculate_cosine_similarity(explanation1, explanation2):
    # Flatten the explanation maps to 1D arrays
    explanation1_flat = explanation1.flatten()
    explanation2_flat = explanation2.flatten()

    # Calculate cosine similarity - the result is a matrix
    similarity_matrix = cosine_similarity([explanation1_flat], [explanation2_flat])

    # The similarity score is the first element of the matrix
    similarity_score = similarity_matrix[0, 0]
    
    return similarity_score


def calculate_euclidean_distance(explanation1, explanation2):
    # Flatten the explanation maps to 1D arrays
    explanation1_flat = explanation1.flatten()
    explanation2_flat = explanation2.flatten()

    # Calculate Euclidean distance
    distance = np.linalg.norm(explanation1_flat - explanation2_flat)
    
    return distance


def l2_normalize(arr):
    norm = np.linalg.norm(arr)
    if norm == 0: 
        return arr
    return arr / norm


def append_results(file_path, new_results):
    # Check if file exists
    if os.path.exists(file_path):
        # Load existing data
        df = pd.read_csv(file_path)
    else:
        # Create a new DataFrame if file doesn't exist
        df = pd.DataFrame()

    # Convert new_results to DataFrame
    new_df = pd.DataFrame(new_results)

    # Append new data
    updated_df = df.concat(new_df, ignore_index=True)

    # Save to CSV
    updated_df.to_csv(file_path, index=False)



