import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

# def calculate_cosine_similarity(explanation1, explanation2):
#     # Flatten the explanation maps to 1D arrays
#     # Calculate cosine similarity - the result is a matrix
#     similarity_matrix = cosine_similarity([explanation1], [explanation2])

#     # The similarity score is the first element of the matrix
#     similarity_score = similarity_matrix[0, 0]
    
#     return similarity_score
def calculate_cosine_similarity(explanation1, explanation2):
    # Flatten the explanation maps to 1D arrays
    explanation1_flat = explanation1.flatten()
    explanation2_flat = explanation2.flatten()

    # Remove NaN values from both arrays

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
    # Convert new_results to DataFrame
    new_df = pd.DataFrame(new_results)

    # # Ensure new_df has the correct shape
    # if not all(isinstance(val, list) for val in new_results.values()):
    #     raise ValueError("All values in new_results must be lists of the same length.")

    # Check if file exists
    if os.path.exists(file_path):
        # Load existing data
        df = pd.read_csv(file_path)
        # Use pd.concat for appending
        updated_df = pd.concat([df, new_df], ignore_index=True)
    else:
        # If file doesn't exist, new_df is the updated DataFrame
        updated_df = new_df

    # Save to CSV
    updated_df.to_csv(file_path, index=False)



