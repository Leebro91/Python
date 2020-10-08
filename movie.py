import time
import operator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

rating_data = pd.io.parsers.read_csv('ratings.dat',
                                     names = ['user_id','movie_id','rating','time'], 
                                     delimiter = "::")

movie_data = pd.io.parsers.read_csv('movies.dat',
                                     names = ['movie_id','title','genre'], 
                                     delimiter = "::")

user_data = pd.io.parsers.read_csv('users.dat',
                                     names = ['user_id','gender','age','occupation','zipcode'], 
                                     delimiter = "::")

rating_data.head()
movie_data.head()
user_data.head()

movie_data.info()

#########################################################
# 분석
#########################################################

# 총 영화 개수
print("Total Movie : ", len(movie_data['movie_id']))

# 총 영화개수(고유)
print("Total Movie : ", len(movie_data['movie_id'].unique()))

# 연도별 영화 개수 많은 10개 연도

movie_data['year'] = movie_data['title'].apply(lambda x:x[-5:-1])
movie_data.head()

movie_data['year'].value_counts().head(10)

from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

