#!/usr/local/bin/python
import pandas as pd
import scipy.sparse as sp

cols = ["user", "item", "rating"]

data = {'user': ['A', 'A', 'A', 'B', 'B', "C", 'C'], 'item': ['X', 'Z', 'W', 'Y', 'W', 'Z', 'Y'], 'rating': [5, 3, 1, 2, 4, 3, 2], 'time': [6, 5, 4, 3, 2, 1, 0]}

data = pd.DataFrame(data)

print(data)


data = data[data["rating"] >= 2]

print(data)

unique_user = data["user"].unique()

print(unique_user)

user2id = pd.Series(data=range(len(unique_user)), index=unique_user)

print(user2id)

data["user"] = data["user"].map(user2id)

print(data)

userids = user2id.to_dict()

print(userids)

unique_item = data["item"].unique()
item2id = pd.Series(data=range(len(unique_item)), index=unique_item)
data["item"] = data["item"].map(item2id)
num_items = len(unique_item)
itemids = item2id.to_dict()

data.sort_values(by=["user", "time"], inplace=True)

print(data)

time_matrix = sp.csr_matrix((data["time"], (data["user"], data["item"])), shape=(3, 4))

print(time_matrix)

user_grouped = data.groupby(by=["user"])
for user, u_data in user_grouped:

    print(user)
    print(u_data)
    u_data = u_data.sample(frac=1)
    print(u_data)
    print("************")

