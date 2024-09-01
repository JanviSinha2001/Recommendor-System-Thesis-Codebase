import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Generate the simulated user-item interaction matrix
def generate_user_item_matrix(data, n_users=1000, seed=42):
    np.random.seed(seed)
    n_items = len(data)
    user_item_matrix = np.random.randint(1, 11, size=(n_users, n_items))
    mask = np.random.rand(n_users, n_items) < 0.8
    user_item_matrix[mask] = 0
    return user_item_matrix, ~mask  # Inverting mask for train/test split

# Split the matrix into train and test sets
def split_data(user_item_matrix, mask):
    train_data = np.copy(user_item_matrix)
    test_data = np.copy(user_item_matrix)
    train_data[mask] = 0  # Simulate unseen interactions in the training data
    test_data[~mask] = 0  # Test data only has the actual interactions
    return train_data, test_data

# Calculate item similarity matrix using cosine similarity
def calculate_item_similarity(train_data):
    return cosine_similarity(train_data.T)

# Predict ratings using Item-Item Collaborative Filtering
def predict_ratings(train_data, item_similarity):
    pred_ratings = item_similarity.dot(train_data.T) / np.array([np.abs(item_similarity).sum(axis=1)]).T
    return pred_ratings.T

# Calculate evaluation metrics
def calculate_metrics(actual_ratings, predicted_ratings):
    mse = mean_squared_error(actual_ratings, predicted_ratings)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_ratings, predicted_ratings)
    nmae = mae / (np.max(actual_ratings) - np.min(actual_ratings))
    return mse, rmse, mae, nmae

# Main function to run the collaborative filtering analysis
def main():
    file_path = '../data/netflix_list.csv'
    data = load_data(file_path)
    
    user_item_matrix, mask = generate_user_item_matrix(data)
    train_data, test_data = split_data(user_item_matrix, mask)
    
    item_similarity = calculate_item_similarity(train_data)
    predicted_ratings = predict_ratings(train_data, item_similarity)
    
    test_mask = test_data > 0
    predicted_test_ratings = predicted_ratings[test_mask]
    actual_test_ratings = test_data[test_mask]
    
    mse, rmse, mae, nmae = calculate_metrics(actual_test_ratings, predicted_test_ratings)
    
    results = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "NMAE"],
        "Value": [mse, rmse, mae, nmae]
    })
    
    print(results)

if __name__ == "__main__":
    main()
