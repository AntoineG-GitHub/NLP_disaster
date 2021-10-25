"""NLP_model file."""
from Dataset import Data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from typing import Dict, Any
from sklearn.metrics import f1_score


class Model():
    """Class used to create the object representing the model."""

    def __init__(self, hyper_parameters: Dict[str, Any]) -> None:
        """Initialize the Object Model with hyper-parameters."""
        self.data = Data()
        self.solver = hyper_parameters["solver"]
        self.max_iter = hyper_parameters["max_iter"]
        self.vectorizer = CountVectorizer()
        self.model_genre = self.model()

    def model(self):
        """
        Use to create and compile the model.

        :return: the compiled model with correponsing hyperparameters
        """
        model = LogisticRegression(solver=self.solver, max_iter=self.max_iter, penalty='l2')
        return model

    def fit_model(self, X_train, y_train) -> None:
        """Use to fit and save the model."""

        x_train = X_train.map(lambda txt: self.data.filter_synopsis(txt))
        X_train_vectorized = self.vectorizer.fit_transform(x_train)
        self.model_genre.fit(X_train_vectorized, y_train)

        pickle.dump(self.model_genre, open('my_ml_model', 'wb'))

    def predict_model(self, x_test) -> np.array:
        """
        Use to predict the genre of the test set movies. First preprocessing the test text and then loading first the model.

        :return: an array of probabilities of prediction
        """
        x_test = x_test.map(lambda txt: self.data.filter_synopsis(txt))
        X = self.vectorizer.transform(x_test)
        self.model_genre = pickle.load(open('my_ml_model', 'rb'))
        prediction = self.model_genre.predict(X)

        return prediction

    def score(self, y_test, prediction):
        """
        score prediction on F1 score
        :param y_test: real data
        :param prediction: predicted data
        :return: score of the model
        """
        score = f1_score(y_test, prediction)
        return score
