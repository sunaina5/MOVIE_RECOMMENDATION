import numpy as np 
import pandas as pd 
import warnings

warnings.filterwarnings('ignore')
columns_name = ["user_id" , "item_id" , "ratings" , "timestamp"]
df = pd.read_csv("ml-100k\ml-100k/u.data" , sep='\t',names= columns_name)
print(df.head())
print(df.shape)

#all the user_id
print(df["user_id"])
#how many unique users
print(df['user_id'].nunique())

#how many unique movies
print(df['item_id'].nunique())

#read now u.item

movies_titles = pd.read_csv("ml-100k\ml-100k/u.item" , sep='\|' , header=None)

print(movies_titles.shape)
movies_titles = movies_titles[[0,1]]
print(movies_titles.head())

movies_titles.columns = ['item_id' , 'title']
print(movies_titles.head())


#merging both u.data (df) and u.item (movies_titles)

df = pd.merge(df, movies_titles, on='item_id')
print(df)

#EXPLORATORY DATA ANALYSIS
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('white')

#Avg rating

print(df.groupby('title').mean())
print(df.groupby('title').mean()['ratings'])

#which movie has the highest rating
print(df.groupby('title').mean()['ratings'].sort_values(ascending = False).head())

#how many no. of rating uses to watch (particular movie)

print(df.groupby('title').count()['ratings'].sort_values(ascending = False))

#create dataframe
ratings=pd.DataFrame(df.groupby('title').mean()['ratings'])
print(ratings)

#how many no. of rating by dataframe added column
ratings['number of ratings']= pd.DataFrame(df.groupby('title').count()['ratings'])
print(ratings)
print(ratings.sort_values(by='ratings',ascending = False))

#histogram of numbers of ratings
plt.figure(figsize=(10,6))
plt.hist(ratings['number of ratings'],bins=70)
print(plt.show())

plt.hist(ratings['ratings'],bins=70)
print(plt.show())

#seaborn --- jointplot
p = sns.jointplot(x ='ratings',y ='number of ratings',data = ratings , alpha=0.5)
print(p)

#CREATING MOVIE RECOMMENDATION

moviematrix = df.pivot_table(index = 'user_id' , columns = 'title', values = 'ratings')
print(moviematrix.head())

#user wise rating
starkid_user_ratings = moviematrix['Star Kid (1997)']
print(starkid_user_ratings.head())

similar_to_starkid = moviematrix.corrwith(starkid_user_ratings)
print(similar_to_starkid)

corr_starkid = pd.DataFrame(similar_to_starkid , columns=['correlation'])
corr_starkid.dropna(inplace = True)
print(corr_starkid)

corr_starkid.sort_values('correlation',ascending = False).head(10)

corr_starkid= corr_starkid.join(ratings['number of ratings'])
print(corr_starkid.head())

corr_starkid = [corr_starkid['number of ratings']>100].sort_values('corrleation', ascending = False)
print(corr_starkid)