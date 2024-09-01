import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    return user_item_matrix

# Perform Matrix Factorization using SVD
def perform_svd(user_item_matrix, n_components=20):
    svd = TruncatedSVD(n_components=n_components)
    latent_matrix = svd.fit_transform(user_item_matrix)
    reconstructed_matrix = svd.inverse_transform(latent_matrix)
    return reconstructed_matrix

# Calculate evaluation metrics
def calculate_metrics(actual_ratings, predicted_ratings):
    mse = mean_squared_error(actual_ratings, predicted_ratings)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_ratings, predicted_ratings)
    nmae = mae / (np.max(actual_ratings) - np.min(actual_ratings))
    return mse, rmse, mae, nmae

# Main function to run matrix factorization
def main():
    file_path = '../data/netflix_list.csv'
    data = load_data(file_path)
    
    user_item_matrix = generate_user_item_matrix(data)
    reconstructed_matrix = perform_svd(user_item_matrix)
    
    actual_ratings = user_item_matrix[user_item_matrix > 0]
    predicted_ratings = reconstructed_matrix[user_item_matrix > 0]
    
    mse, rmse, mae, nmae = calculate_metrics(actual_ratings, predicted_ratings)
    
    results = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "NMAE"],
        "Value": [mse, rmse, mae, nmae]
    })
    
    print(results)

if __name__ == "__main__":
    main()
