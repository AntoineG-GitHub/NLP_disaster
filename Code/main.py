from NLP_model import Model
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Create train data and validation data
    x_train_data = train_data.text
    y_train_data = train_data.target
    X_train, X_test, y_train, y_test = train_test_split(x_train_data, y_train_data, test_size=0.33, random_state=42)

    hyper_parameters = {"solver": "sag", "max_iter": 100}

    model_valid = Model(hyper_parameters)
    model_valid.fit_model(X_train, y_train)
    prediction = model_valid.predict_model(X_test)
    score = model_valid.score(y_test, prediction)
    print(score)

    # Train and predict on full data and test data
    model_test = Model(hyper_parameters)
    model_test.fit_model(x_train_data, y_train_data)
    prediction_test = pd.DataFrame(model_test.predict_model(X_test))
    prediction_test.to_csv("submission.csv")


if __name__ == "__main__":
    train_data = pd.read_csv("D:/Users/Antoine/Documents/Programmation_Projects/NLP_disaster/Data/train.csv")
    test_data = pd.read_csv("D:/Users/Antoine/Documents/Programmation_Projects/NLP_disaster/Data/test.csv")
    main()
