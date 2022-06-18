from saver import Saver
from plotter import plotter
from movies import MovieRatings
from recommender import Recommender

#import datasets
movies = MovieRatings('./Datasets/tmdb_5000_credits.csv', './Datasets/tmdb_5000_movies.csv')

#print the first five info to see what it is like
# movies.getData(5)

#get the mean of the vote average
C = movies.get_mean_vote_ratings()
# c

#So, the mean rating for all the movies is approx 6 on a scale of 10.The next step is to determine an appropriate value for m, the minimum votes required to be listed in the chart. We will use 90th percentile as our cutoff. In other words, for a movie to feature in the charts, it must have more votes than at least 90% of the movies in the list.

m = movies.get_percentile(0.9)
# m

#calculate the vote_count that are more or equal to the quantile
q_movies = movies.get_list_by_quantile(m)
# q_movies.shape

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(movies.weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

# Print the top 15 movies
# print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))


# We have made our first(though very basic) recommender. 
# Under the Trending Now tab of these systems we find movies that are very popular and 
# they can just be obtained by sorting the dataset by the popularity column.

allMovies = movies.get_all_data();
pop = allMovies.sort_values('popularity', ascending=False)

plt = plotter(12,4)
plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center', color='blue')
plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.savefig('foo.png',dpi=400)

#using seaborn to visualize the same data
plt.set_seaborn()
plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center', color='blue')

plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.savefig('seaborn.png', dpi=400)

recommender = Recommender(allMovies, "overview", "title")
recommender.set_features(['cast', 'crew', 'keywords', 'genres'])

recommender.define_features(['cast', 'keywords', 'genres'])
recommender.apply_clean_data(['cast', 'keywords', 'director', 'genres'])

recommender.get_data()
original_stdout = recommender.get_output()

list = recommender.get_recommendations('Iron Man')


saver = Saver(list, original_stdout)
saver.saveCSV("recommendation.csv")
saver.saveHDF5("recommendation.hdf5")
    