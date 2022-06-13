import pandas as pd 
import numpy as np 

#import datasets
df1=pd.read_csv('./Datasets/tmdb_5000_credits.csv')
df2=pd.read_csv('./Datasets/tmdb_5000_movies.csv')

#merge the datasets
df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')

#print the first five info to see what it is like
# df2.head(5)

#get the mean of the vote average
C = np.mean(df2['vote_average'])
# c

m = np.quantile(df2['vote_count'], 0.9)
# m

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)
