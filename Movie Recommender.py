#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
from collections import defaultdict


# In[18]:


# Load the dataset
data = Dataset.load_builtin('ml-100k')

#Define Format type
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file('Users/michaelamcnair/Desktop/ml-100k/u.data', reader=reader)


# In[7]:


# Split the dataset for 5-fold cross-validation
from surprise.model_selection import train_test_split
trainset, testset = train_test_split(data, test_size=0.25)


# In[10]:


# Use KNNBasic algorithm
algo = KNNBasic()


# In[11]:


algo.fit(trainset)


# In[13]:


from surprise import SVD, Dataset, accuracy
from surprise.model_selection import train_test_split, cross_validate


# In[14]:


# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Or evaluate on the test set
predictions = algo.test(testset)
accuracy.rmse(predictions)


# In[20]:


from collections import defaultdict


# In[25]:


def get_top_n(predictions, n=10):
    """Return the top N recommendations for each user from a set of predictions."""
    
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the N highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Use this function after making predictions to get the top-N recommendations for each user
top_n = get_top_n(predictions, n=10)
# Now you can print or analyze the recommended items for each user


# In[26]:


print(top_n)


# In[27]:


for uid, user_ratings in top_n.items():
    print(f"User {uid}:")
    for iid, rating in user_ratings:
        print(f"\tItem {iid} with estimated rating of {rating:.2f}")


# In[31]:


user_id = '1'  # Adjust based on your dataset
top_n = get_top_n(predictions, n=10)

# Extract item IDs and their estimated ratings for the user
items, ratings = zip(*top_n[user_id])

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(items, ratings, color='green')
plt.xlabel('Estimated Rating')
plt.ylabel('Item ID')
plt.title(f'Top 10 Recommendations for User {user_id}')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest rating at the top
plt.show()


# In[ ]:




