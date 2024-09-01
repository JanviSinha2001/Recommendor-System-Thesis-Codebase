import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Compute the TF-IDF matrix for the plot descriptions
def compute_tfidf_matrix(data):
    tfidf = TfidfVectorizer(stop_words='english')
    data['plot'] = data['plot'].fillna('')
    tfidf_matrix = tfidf.fit_transform(data['plot'])
    return tfidf_matrix

# Compute the cosine similarity matrix
def compute_cosine_similarity(tfidf_matrix):
    return linear_kernel(tfidf_matrix, tfidf_matrix)

# Generate content-based recommendations for a given item
def get_recommendations(title, data, cosine_sim):
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get the scores of the 10 most similar items

    item_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[item_indices]

# Main function to run content-based filtering
def main():
    file_path = '../data/netflix_list.csv'
    data = load_data(file_path)
    
    tfidf_matrix = compute_tfidf_matrix(data)
    cosine_sim = compute_cosine_similarity(tfidf_matrix)
    
    title = 'Lucifer'  #title to get recommendations for
    recommendations = get_recommendations(title, data, cosine_sim)
    
    print(f"Recommendations for '{title}':\n")
    print(recommendations)

if __name__ == "__main__":
    main()
