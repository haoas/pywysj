import os
import pandas as pd
from collections import defaultdict
from operator import itemgetter

datafile = "D:/Pythonsjwjrm/samplecode/ml-100k/u.data"
datepare = lambda x: pd.to_datetime(x, unit='s')
all_ratings = pd.read_csv(datafile, delimiter="\t", header=None, date_parser=datepare,
                          names=["UserID", "MovieID", "Rating", "Datetime"])
all_ratings["Datetime"] = pd.to_datetime(all_ratings['Datetime'], unit='s')

all_ratings["Favorable"] = all_ratings["Rating"] > 3

ratings = all_ratings[all_ratings["UserID"].isin(range(30))]
favorable_ratings = ratings[ratings["Favorable"] == True]

favorable_reviews_by_users = dict((k, frozenset(v.values)) for k, v in favorable_ratings.groupby("UserID")["MovieID"])

x = ratings[["MovieID", "Favorable"]]
num_favorable_by_movie = x.groupby("MovieID").sum()

frequent_itemsets = {}
min_support = 10

frequent_itemsets[1] = dict(
    (frozenset((movie_id,)), row["Favorable"]) for movie_id, row in num_favorable_by_movie.iterrows() if
    row["Favorable"] > min_support)


# print(num_favorable_by_movie.sort_values("Favorable",ascending=False))

def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other_reviewd_movie in reviews - itemset:
                    current_supperset = itemset | frozenset((other_reviewd_movie,))
                    counts[current_supperset] += 1
    return dict((itemset, frequent) for itemset, frequent in counts.items() if frequent > min_support)


for k in range(2, 20):
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users, frequent_itemsets[k - 1], min_support)
    frequent_itemsets[k] = cur_frequent_itemsets

del frequent_itemsets[1]

candidate_rulers = []
for itemset_length, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset:
            premise = itemset - set({conclusion, })
            candidate_rulers.append((premise, conclusion))

correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, review in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rulers:
        permise, conclusion = candidate_rule
        if permise.issubset(review):
            if conclusion in review:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1

filter_correct_counts = {k: v for k, v in correct_counts.items() if v > 1}

rule_confidence = {condidate_rule: filter_correct_counts[condidate_rule] / float( filter_correct_counts[condidate_rule] + incorrect_counts[condidate_rule]) for condidate_rule in candidate_rulers}

sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)

print(favorable_reviews_by_users.items())

for index in range(50):
    print("Rule #{0}:".format(index + 1))
    (permise, conclusion) = sorted_confidence[index][0]
    print("IF a persion recommends  {0} ,they  will alseo recommends {1} by probability {2:.3f} ".format(permise,conclusion, sorted_confidence[index][1]))
    print("--Confidence {0:.3f}".format(rule_confidence[(permise, conclusion)]))
