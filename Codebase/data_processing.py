import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = '../data/netflix_list.csv'  # Adjust the path to match your folder structure
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Displaying the first few rows of the dataset:")
print(data.head())

# Check for missing values and handle them
print("Checking for missing values:")
print(data.isnull().sum())

# Example: Fill missing values with the mean of the column
data.fillna(data.mean(), inplace=True)
print("Missing values filled with mean values.")

# Encode categorical variables
label_encoder = LabelEncoder()
data['categorical_column'] = label_encoder.fit_transform(data['categorical_column'])
print("Categorical column encoded.")

# Standardize numerical columns
scaler = StandardScaler()
data[['numerical_column1', 'numerical_column2']] = scaler.fit_transform(data[['numerical_column1', 'numerical_column2']])
print("Numerical columns standardized.")

# Save the preprocessed data
preprocessed_file_path = '../data/preprocessed_netflix_list.csv'
data.to_csv(preprocessed_file_path, index=False)
print("Preprocessed data saved to", preprocessed_file_path)
