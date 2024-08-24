import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item interaction data (e.g., movie ratings)
user_item_data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
    'item_id': [1, 2, 3, 1, 2, 2, 3, 3, 4, 4],
    'rating': [5, 4, 1, 4, 5, 3, 4, 2, 3, 5]
}
user_item_df = pd.DataFrame(user_item_data)

# Create a User-Item Matrix
user_item_matrix = user_item_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Apply Singular Value Decomposition (SVD) for Collaborative Filtering
svd = TruncatedSVD(n_components=2)
matrix = user_item_matrix.values
U = svd.fit_transform(matrix)
sigma = svd.singular_values_
Vt = svd.components_

# Reconstruct the matrix
reconstructed_matrix = np.dot(np.dot(U, np.diag(sigma)), Vt)

# Convert back to DataFrame for interpretability
reconstructed_df = pd.DataFrame(reconstructed_matrix, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Sample item data with descriptions
item_data = {
    'item_id': [1, 2, 3, 4],
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
    'description': [
        'A thrilling action movie with lots of suspense.',
        'A romantic comedy about a couple finding love.',
        'A sci-fi film set in a futuristic world.',
        'An epic adventure with a heroic quest.'
    ]
}
item_df = pd.DataFrame(item_data)

# Create TF-IDF vectors for item descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(item_df['description'])

# Compute cosine similarity matrix for content-based filtering
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Combined Recommendation Function
def recommend_items(user_id, item_title=None, num_recommendations=3):
    # Collaborative Filtering Recommendations
    user_ratings = reconstructed_df.loc[user_id]
    cf_recommendations = user_ratings.sort_values(ascending=False).head(num_recommendations).index

    # Content-Based Filtering Recommendations
    if item_title:
        indices = pd.Series(item_df.index, index=item_df['title']).drop_duplicates()
        idx = indices[item_title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]
        content_recommendations = [item_df['title'].iloc[i[0]] for i in sim_scores]
    else:
        content_recommendations = []

    return cf_recommendations.tolist(), content_recommendations

# Example usage
user_id = 1
item_title = 'Movie A'  # Set to None if you don't want content-based recommendations
cf_recommendations, content_recommendations = recommend_items(user_id, item_title)

print(f"Collaborative Filtering recommendations for User {user_id}: {cf_recommendations}")
print(f"Content-Based Filtering recommendations for '{item_title}': {content_recommendations}")
