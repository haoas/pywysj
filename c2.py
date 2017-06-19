import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def main():
    datepare = lambda x: pd.to_datetime(x)
    dataset = pd.read_csv('d:/nbascore.csv', parse_dates=[0], date_parser=datepare, skiprows=[0, ])
    dataset.columns = ["Date", "Start", "Visitor Team", "VisitorPts", "Home Team", "HomePts", "Score Type", "OT",
                       "Notes"]

    dataset['HomeWin'] = dataset["VisitorPts"] < dataset["HomePts"]
    y_true = dataset["HomeWin"].values
    dataset["HomeLastWin"] = False
    dataset["VisitorLastWin"] = False
    won_last = defaultdict(int)
    for index, row in dataset.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        row["HomeLastWin"] = won_last[home_team]
        row["VisitorLastWin"] = won_last[visitor_team]
        dataset.ix[index] = row
        won_last[home_team] = row["HomeWin"]
        won_last[visitor_team] = not row["HomeWin"]
        # print(dataset[:5])
    # print(won_last["Los Angeles Clippers"])

    clf = DecisionTreeClassifier(random_state=14)
    X_previouswin = dataset[["HomeLastWin", "VisitorLastWin"]].values
    scores = cross_val_score(clf, X_previouswin, y_true, scoring="accuracy")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

    encoding = LabelEncoder()
    encoding.fit(dataset["Home Team"].values)
    home_teams = encoding.transform(dataset["Home Team"].values)
    visitor_teams = encoding.transform(dataset["Visitor Team"].values)
    X_teams = np.vstack([home_teams, visitor_teams]).T

    onehot = OneHotEncoder()
    X_teams_ex = onehot.fit_transform(X_teams).todense()

    clf1 = DecisionTreeClassifier(random_state=14)
    scores1 = cross_val_score(clf1, X_teams_ex, y_true, scoring='accuracy')
    print("Accuracy: {0:.1f}%".format(np.mean(scores1) * 100))

    clf2 = RandomForestClassifier(random_state=14)
    scores = cross_val_score(clf2, X_teams_ex, y_true, scoring='accuracy')
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


main()
