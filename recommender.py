
import sys
import numpy as np
import pandas as pd

from ast import literal_eval
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class Recommender:
    def __init__(self, movies, transformField, useField):
        tfidf = TfidfVectorizer(stop_words='english')
        movies[transformField] = movies[transformField].fillna('')

        self.list_of_all_titles = movies[useField].tolist()
        tfidf_matrix = tfidf.fit_transform(movies['overview'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
        
        self.movies = movies

    def get_recommendations(self, title):
        # Get the index of the movie that matches the title
        idx = self.indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return self.movies['title'].iloc[movie_indices]
    
    def set_features (self, list):
        for feature in list:
            self.movies[feature] = self.movies[feature].apply(literal_eval)
    
    # Get the director's name from the crew feature. If director is not listed, return NaN
    def set_director(self, x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    # Returns the list top 3 elements or entire list; whichever is more.
    def get_list(self, x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > 3:
                names = names[:3]
            return names

        #Return empty list in case of missing/malformed data
        return []

    # Function to convert all strings to lower case and strip names of spaces
    def clean_data(self, x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            #Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    # Define new director, cast, genres and keywords features that are in a suitable form.
    def define_features(self, features):
        self.movies['director'] = self.movies['crew'].apply(self.set_director)
        for feature in features:
            self.movies[feature] = self.movies[feature].apply(self.get_list)
    
    def apply_clean_data(self, features):
        for feature in features:
            self.movies[feature] = self.movies[feature].apply(self.clean_data)
    
    #We can now create our "metadata soup", which is a string that contains all the metadata that we want to feed to our vectorizer (namely actors, director and keywords).
    def create_soup(self, x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

    def get_data(self):
        return self.movies.apply(self.create_soup, axis=1)
        
    def get_output(self):
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(self.get_data())

        # Compute the Cosine Similarity matrix based on the count_matrix

        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)

        # Reset index of our main DataFrame and construct reverse mapping as before
        df2 = self.movies.reset_index()
        self.indices = pd.Series(df2.index, index=df2['title'])

        original_stdout = sys.stdout # Save a reference to the original standard output
        
        return original_stdout