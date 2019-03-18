import pandas as pd
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class BayesianClassifier:

    def __init__(self, X, y, categories):
        self.X = X
        self.y = y
        self.categories = categories

        self.gathered_x_vals = self._gather_x_vals()
        print("Calculating mean and covariance matrices for each category...\n")
        self.means, self.covs = zip(*[(i.mean(axis=0), np.cov(i.T)) for i in self.gathered_x_vals])
        self.priors = self._calculate_priors()

    def _gather_x_vals(self):
        print("Gathering combined x values for each category...\n")
        return [self.X[self.y == cat] for cat in self.categories]

    # Gather observation equal to cat as a proportion of all observations
    def _calculate_priors(self):
        print("Calculating priors...\n")
        priors = []
        for cat in self.categories:
            priors.append(len(self.y[self.y == cat]) / len(self.y))
        return priors

    def sample_y(self, category):
        print("Sampling from a multivariate normal distribution\n")
        return mvn.rvs(mean=self.means[category], cov=self.covs[category])

    def sample(self):
        random_cat = np.random.randint(0, len(self.categories))
        return self.sample_y(random_cat)


if __name__ == "__main__":
    print("Loading csv...")
    dataframe = pd.read_csv("../datasets/mnist_digits.csv")
    print("{} rows found".format(len(dataframe)))

    X_train = dataframe.iloc[:, 1:].values
    y_train = dataframe.iloc[:, 0].values
    categories = sorted(set(y_train))
    print("{} categories found\n".format(len(categories)))

    classifier = BayesianClassifier(X_train, y_train, categories)

    for cat in categories:
        sample = classifier.sample_y(cat).reshape(28, 28)
        cat_means = classifier.means[cat].reshape(28, 28)

        plt.subplot(1, 2, 1)
        plt.imshow(sample, cmap='bone')
        plt.title("Sample")

        plt.subplot(1, 2, 2)
        plt.imshow(cat_means, cmap='bone')
        plt.title("Mean")
        plt.show()

    # Generating a random sample
    random_sample = classifier.sample().reshape(28, 28)
    plt.subplot(1, 2, 1)
    plt.imshow(random_sample, cmap='bone')
    plt.title("Random sample from a Random Class")
    plt.show()
