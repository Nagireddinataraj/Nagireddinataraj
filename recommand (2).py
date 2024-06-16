import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to recommend articles
def recommend():
    # Load articles dataset
    articles_df = pd.read_csv('shared_articles.csv')

    # Filter articles dataset to only include shared content
    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']

    # Load user interactions dataset
    interactions_df = pd.read_csv('users_interactions.csv')

    # Data exploration
    print("Interactions DataFrame Shape:", interactions_df.shape)
    print(interactions_df.head())

    # Merge articles and interactions datasets
    metadata = articles_df.merge(interactions_df.set_index('contentId'), on='contentId', how="inner")

    # Print metadata information for verification
    print("Metadata Length:", len(metadata))
    print(metadata.head())
    print("Metadata Columns:", metadata.columns)

    # Calculate TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(articles_df['text'].fillna(''))  # Fill NaN with empty string

    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Construct a reverse map of indices and article titles
    articles_df = articles_df.reset_index()
    indices = pd.Series(articles_df.index, index=articles_df['title']).drop_duplicates()

    # Get recommendations for a specific article
    print(get_recommendations('The Rise And Growth of Ethereum Gets Mainstream Coverage', indices, cosine_sim, articles_df))


def get_recommendations(title, indices, cosine_sim, data):
    if title not in indices:
        print("Title not found in the dataset.")
        return []
    
    # Get the index of the article that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all articles with that article
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the articles based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar articles
    sim_scores = sim_scores[1:11]

    # Get the article indices
    article_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar article titles
    return data['title'].iloc[article_indices].tolist()


if __name__ == '__main__':
    recommend()
