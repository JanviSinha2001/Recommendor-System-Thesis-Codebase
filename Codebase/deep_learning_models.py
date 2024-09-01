import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
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

# Build and train a simple neural collaborative filtering model
def build_and_train_model(user_item_matrix, n_factors=50, epochs=10):
    n_users, n_items = user_item_matrix.shape
    model = tf.keras.Sequential([
        layers.Embedding(input_dim=n_users, output_dim=n_factors),
        layers.Embedding(input_dim=n_items, output_dim=n_factors),
        layers.Flatten(),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    users = np.repeat(np.arange(n_users), n_items)
    items = np.tile(np.arange(n_items), n_users)
    ratings = user_item_matrix.flatten()
    
    mask = ratings > 0
    users, items, ratings = users[mask], items[mask], ratings[mask]
    
    model.fit([users, items], ratings, epochs=epochs, verbose=1)
    return model

# Predict ratings
def predict_ratings(model, n_users, n_items):
    users = np.repeat(np.arange(n_users), n_items)
    items = np.tile(np.arange(n_items), n_users)
    predicted_ratings = model.predict([users, items])
    return predicted_ratings.reshape((n_users, n_items))

# Calculate evaluation metrics
def calculate_metrics(actual_ratings, predicted_ratings):
    mse = mean_squared_error(actual_ratings, predicted_ratings)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_ratings, predicted_ratings)
    nmae = mae / (np.max(actual_ratings) - np.min(actual_ratings))
    return mse, rmse, mae, nmae

# Main function to run deep learning model
def main():
    file_path = '../data/netflix_list.csv'
    data = load_data(file_path)
    
    user_item_matrix = generate_user_item_matrix(data)
    model = build_and_train_model(user_item_matrix)
    
    predicted_ratings = predict_ratings(model, *user_item_matrix.shape)
    actual_ratings = user_item_matrix[user_item_matrix > 0]
    predicted_ratings = predicted_ratings[user_item_matrix > 0]
    
    mse, rmse, mae, nmae = calculate_metrics(actual_ratings, predicted_ratings)
    
    results = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "NMAE"],
        "Value": [mse, rmse, mae, nmae]
    })
    
    print(results)

if __name__ == "__main__":
    main()
