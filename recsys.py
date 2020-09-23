import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from scipy.linalg import svdvals
from scipy.sparse.linalg import svds
from scipy.spatial.distance import correlation
from surprise.model_selection.split import train_test_split
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.knns import KNNBasic, KNNWithMeans
from surprise.prediction_algorithms.matrix_factorization import SVD, NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF



# Loading relevant csv files
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Plotting freq of ratings

bins = np.linspace(0,5,num=11)
plt.hist(ratings['rating'],bins=bins); plt.xlabel("Rating score"); plt.ylabel("Frequency"); plt.title("Frequency of rating values"); plt.show()


# Plotting number of ratings for each movie

counter = Counter(ratings['movieId'])
num_ratings = sorted(list(counter.values()),reverse=True)
plt.plot(num_ratings); plt.xlabel("Movie index"); plt.ylabel("Number of ratings"); plt.title("Number of ratings for each movie index"); plt.show()


# kNN predictions

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

neighbors = np.linspace(1,101,num=51,dtype=int)
basic_pearson, basic_cosine = [], []
for i in neighbors:
	print(i)
	cv_pearson = cross_validate(KNNBasic(k=i,sim_options={'name':'pearson'},verbose=False), data, cv=5)
	basic_pearson.append(np.mean(cv_pearson['test_rmse']))
	cv_cosine = cross_validate(KNNBasic(k=i,sim_options={'name':'cosine'},verbose=False), data, cv=5)
	basic_cosine.append(np.mean(cv_cosine['test_rmse']))

means_pearson, means_cosine = [], []
for i in neighbors:
	print(i)
	cv_pearson = cross_validate(KNNWithMeans(k=i,sim_options={'name':'pearson'},verbose=False), data, cv=5)
	means_pearson.append(np.mean(cv_pearson['test_rmse']))
	cv_cosine = cross_validate(KNNWithMeans(k=i,sim_options={'name':'cosine'},verbose=False), data, cv=5)
	means_cosine.append(np.mean(cv_cosine['test_rmse']))

fig, ax = plt.subplots()
ax.plot(neighbors,basic_cosine, 'r', label='Cosine')
ax.plot(neighbors, basic_pearson, 'b', label='Pearson')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("5-fold average RMSE"); plt.title("k-NN with 5-fold CV")

fig, ax = plt.subplots()
ax.plot(neighbors,means_cosine, 'r', label='Cosine')
ax.plot(neighbors, means_pearson, 'b', label='Pearson')
ax.legend(loc='best')
plt.xlabel("k"); plt.ylabel("5-fold average RMSE"); plt.title("Mean-centered k-NN with 5-fold CV")
plt.show()


# Make kNN movie recommendations

R_zero = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)

movie_name = "Iron Man (2008)"
movie_ind = movies[movies['title']==movie_name]['movieId']
movie_vec = R_zero.loc[movie_ind,:]

knn = NearestNeighbors(metric='correlation',n_neighbors=37)
R_c = R_zero.values - np.mean(R_zero.values,axis=1).reshape(-1,1)
 knn.fit(R_c)

dist, rec_ind = knn.kneighbors(movie_vec,n_neighbors=11)
rec_movie_ind = R_zero.index[rec_ind][0]

print("Top 10 movie recommendations for ", movie_name, ":")
j = 1
for i in rec_movie_ind[1:]:
	print(str(j)+'.',movies[movies['movieId']==i]['title'].values[0],', Genre = ',movies[movies['movieId']==i]['genres'].values[0])
	j+=1


# SVD singular values

R_zero = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0).values
R_zero = R_zero - np.mean(R_zero,axis=1).reshape(-1,1)
sig = svdvals(R_zero)
plt.plot(sig); plt.xlabel('Index of singular value'); plt.ylabel('Magnitude of singular value'); plt.title('Distribution of singular values'); plt.show()


# SVD predictions

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

rank_k = np.linspace(1,100,num=100,dtype=int)
svd_rmse = []
for i in rank_k:
	print(i)
	cv_svd = cross_validate(SVD(n_factors=i,verbose=False), data, cv=5)
	svd_rmse.append(np.mean(cv_svd['test_rmse']))
	
plt.plot(rank_k,svd_rmse); plt.xlabel("rank k"); plt.ylabel("5-fold average RMSE"); plt.title("SVD with 5-fold CV"); plt.show()


# SVD movie recommendations

R_zero = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0).values
R_centered = R_zero - np.mean(R_zero,axis=1).reshape(-1,1)
U, sigma, Vt = svds(R_centered, k = 6)
sigma = np.diag(sigma)
r_hat = np.dot(np.dot(U, sigma), Vt) + np.mean(R_zero,axis=1).reshape(-1,1)
preds_df = pd.DataFrame(r_hat, columns = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0).columns)
userID = 219#28
user_row_number = userID - 1 # UserID starts at 1, not 0
sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
user_data = ratings[ratings['userId'] == (userID)]
user_full = (user_data.merge(movies, how = 'left', left_on = 'movieId', right_on = 'movieId').sort_values(['rating'], ascending=False))
recommendations = (movies[~movies['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:10, :-1])

movie_name = "Iron Man (2008)"
print("Top 10 movie recommendations for ", movie_name, ":")
j = 1
for i in recommendations[['title','genres']].values:
	print(str(j)+'.',i[0],', Genre = ',i[1])
	j+=1


# NMF predictions

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

rank_k = np.linspace(1,80,num=80,dtype=int)
nmf_rmse = []
for i in rank_k:
	print(i)
	cv_nmf = cross_validate(NMF(n_factors=i,verbose=False), data, cv=5)
	nmf_rmse.append(np.mean(cv_nmf['test_rmse']))
	
plt.plot(rank_k,nmf_rmse); plt.xlabel("rank k"); plt.ylabel("5-fold average RMSE"); plt.title("NNMF with 5-fold CV"); plt.show()


# NMF recommendations
R_zero = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0).values
model = NMF(n_components=17)
W = model.fit_transform(R_zero)
H = model.components_
r_hat = np.dot(W, H)
preds_df = pd.DataFrame(r_hat, columns = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0).columns)
userID = 219#28
user_row_number = userID - 1 # UserID starts at 1, not 0
sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
user_data = ratings[ratings['userId'] == (userID)]
user_full = (user_data.merge(movies, how = 'left', left_on = 'movieId', right_on = 'movieId').sort_values(['rating'], ascending=False))
recommendations = (movies[~movies['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:10, :-1])

movie_name = "Iron Man (2008)"
print("Top 10 movie recommendations for ", movie_name, ":")
j = 1
for i in recommendations[['title','genres']].values:
	print(str(j)+'.',i[0],', Genre = ',i[1])
	j+=1

