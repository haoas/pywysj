import numpy as np
from sklearn.datasets import load_iris
from collections import defaultdict
from operator import itemgetter
from sklearn.model_selection import train_test_split


class Test():
    def main(self):
        # x=np.loadtxt('c:\LogFile.txt')
        dataset = load_iris()
        x = dataset.data
        y = dataset.target
        attribute_means = x.mean(axis=0)
        X_d = np.array(x >= attribute_means, dtype='int')
        # print(attribute_means)
        # print(X_d)
        Xd_train, Xd_test, y_train, y_test = train_test_split(X_d, y, random_state=14)
        # print(Xd_train)
        # print(Xd_train.shape[1])
        all_predictors = {}
        errors = {}
        for feature_index in range(Xd_train.shape[1]):
            predictors, total_error = self.train_on_feature(Xd_train, y_train, feature_index)
            all_predictors[feature_index] = predictors
            errors[feature_index] = total_error
        best_feature, best_error = sorted(errors.items(), key=itemgetter(1))[0]
        model = {'feature': best_feature, 'predictor': all_predictors[best_feature]}
        y_predicted = self.predict(Xd_test, model)

        print(y_predicted)
        accuracy = np.mean(y_predicted == y_test) * 100
        print("The test accuracy is {:.1f}%".format(accuracy))

    def predict(self, X_test, model):
        variable = model['feature']
        predictor = model['predictor']
        y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
        print(model)
        print(variable)
        print(predictor)
        return y_predicted

    def train_feature_value(self, x, y_feature, feature_index, value):
        class_counts = defaultdict(int)
        for sample, y in zip(x, y_feature):
            if sample[feature_index] == value:
                class_counts[y] += 1
        sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
        most_frequent_class = sorted_class_counts[0][0]

        incorrect_predictions = [class_count for class_value, class_count in class_counts.items() if
                                 class_value != most_frequent_class]
        error = sum(incorrect_predictions)

        return most_frequent_class, error

    def train_on_feature(self, X, y_true, feature_index):
        values = set(X[:, feature_index])
        predictors = {}
        errors = []
        for current_value in values:
            most_frequent_class, error = self.train_feature_value(X, y_true, feature_index, current_value)
            predictors[current_value] = most_frequent_class
            errors.append(error)
        total_error = sum(errors)
        return predictors, total_error


t = Test()
t.main()