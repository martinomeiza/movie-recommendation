import pandas as pd 
import h5py
import sys
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from weight_rank import weighted_rating

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

#So, the mean rating for all the movies is approx 6 on a scale of 10.The next step is to determine an appropriate value for m, the minimum votes required to be listed in the chart. We will use 90th percentile as our cutoff. In other words, for a movie to feature in the charts, it must have more votes than at least 90% of the movies in the list.

m = np.quantile(df2['vote_count'], 0.9)
# m

#calculate the vote_count that are more or equal to the quantile
q_movies = df2.copy().loc[df2['vote_count'] >= m]
# q_movies.shape


# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
# q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)


#We have made our first(though very basic) recommender. 
#Under the Trending Now tab of these systems we find movies that are very popular and 
#they can just be obtained by sorting the dataset by the popularity column.
pop= df2.sort_values('popularity', ascending=False)
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='blue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.savefig('foo.png',dpi=400)

#using seaborn to visualize the same data
import seaborn as sns
sns.set()
# same plotting code as above!
plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='blue')

plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.savefig('seaborn.png',dpi=400)

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])
#We see that over 20,000 different words were used to describe the 4800 movies in our dataset.
#With this matrix in hand, we can now compute a similarity score.


# creating a list with all the movie names given in the dataset

list_of_all_titles = df2['title'].tolist()

#We will be using the cosine similarity to calculate a numeric quantity that denotes the similarity between two movies. 

# Compute the cosine similarity matrix
#Since we have used the TF-IDF vectorizer, calculating the dot product will directly 
#give us the cosine similarity score. Therefore, we will use sklearn's linear_kernel() 
#instead of cosine_similarities() since it is faster.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies

#We are now in a good position to define our recommendation function. These are the following steps we'll follow :-

# Get the index of the movie given its title.
# Get the list of cosine similarity scores for that particular movie with all movies. 
# Convert it into a list of tuples where the first element is its position and the second is the similarity score.
# Sort the aforementioned list of tuples based on the similarity scores; that is, the second element.
# Get the top 10 elements of this list. Ignore the first element as it refers to self (the movie most similar to a particular movie is the movie itself).
# Return the titles corresponding to the indices of the top elements.


def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]

# Parse the stringified features into their corresponding python objects
from ast import literal_eval


#We are going to build a recommender based on the following metadata: 
# the 3 top actors, the director, related genres and the movie plot keywords.


features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)
    
# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Define new director, cast, genres and keywords features that are in a suitable form.
df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)
    
    
#The next step would be to convert the names and keyword instances into lowercase and strip all the spaces between them. This is done so that our vectorizer doesn't count the Chris of "Chris Rock" and "Chris Helmsworth" as the same.

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)
    
#We can now create our "metadata soup", which is a string that contains all the metadata that we want to feed to our vectorizer (namely actors, director and keywords).
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)



#The next steps are the same as what we did with our plot description based recommender
#the main difference is that we use the CountVectorizer() instead of TF-IDF. 
#This is because we do not want to down-weight the presence of an actor/director if he or she has acted or directed in relatively more movies.
# Import CountVectorizer and create the count matrix

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of our main DataFrame and construct reverse mapping as before
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])

original_stdout = sys.stdout # Save a reference to the original standard output

with open('recommendation.csv', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    outp = get_recommendations('Iron Man', cosine_sim2)
    print(outp)
    sys.stdout = original_stdout # Reset the standard output to its original value
    
with h5py.File('recommendation.hdf5', 'w') as f:
    outp = get_recommendations('Iron Man', cosine_sim2)
    dset = f.create_dataset("default", data = outp)
    
# with h5py.File('recommendation.hdf5', 'w') as f:
# #     dset = f.create_dataset("default", data = arr)
#     sys.stdout = f # Change the standard output to the file we created.
#     outp = get_recommendations('Iron Man', cosine_sim2)
#     print(outp)
#     sys.stdout = original_stdout
    