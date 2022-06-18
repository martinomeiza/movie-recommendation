

import numpy as np 
import pandas as pd 

class MovieRatings:
    def __init__(self, moviesCSV, creditsCSV):
        df1=pd.read_csv('./Datasets/tmdb_5000_credits.csv')
        df2=pd.read_csv('./Datasets/tmdb_5000_movies.csv')
        
        df1.columns = ['id','tittle','cast','crew']
        self.movies= df2.merge(df1,on='id')
        
    def get_all_data(self):
        return self.movies
        
    def get_data(self, count):
        return self.movies.head(count)
        
    def get_mean_vote_ratings(self):
        return np.mean(self.movies['vote_average'])
    
    def get_percentile(self, percentile):
        self.percentile = percentile
        return np.quantile(self.movies['vote_count'], percentile)
        
    def get_list_by_quantile(self, quantile):
        return self.movies.copy().loc[self.movies['vote_count'] >= quantile]
    
    def weighted_rating(self, list):
        v = list['vote_count']
        R = list['vote_average']
        
        c = self.get_mean_vote_ratings()
        m = self.get_percentile(self.percentile)
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * c)