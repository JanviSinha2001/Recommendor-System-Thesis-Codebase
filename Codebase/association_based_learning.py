import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset

def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess data for association rule learning

def preprocess_data(data):
    # Let's assume genres can be used for association rule mining
    data = data.copy()
    data['genres'] = data['genres'].apply(lambda x: x.split(','))
    data = data.explode('genres')

    # Create a one-hot encoded dataframe for genres
    genre_encoded = pd.get_dummies(data['genres'])
    return genre_encoded

# Perform association rule mining using the Apriori algorithm

def generate_association_rules(genre_encoded, min_support=0.01, metric="lift", min_threshold=1):
    frequent_itemsets = apriori(
        genre_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(
        frequent_itemsets, metric=metric, min_threshold=min_threshold)
    return rules

# Main function to run association-based learning

def main():
    file_path = '../data/netflix_list.csv'
    data = load_data(file_path)

    genre_encoded = preprocess_data(data)
    rules = generate_association_rules(genre_encoded)

    print("Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


if __name__ == "__main__":
    main()
