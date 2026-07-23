from text_classifier.data.model import get_model_data


def main():
    _, _, X_train, X_test, y_train, y_test = get_model_data()
    print(f"X_train: \n{X_train.head()}\n")
    print(f"X_test: \n{X_test.head()}\n")
    print(f"y_train: \n{y_train}\n")
    print(f"y_test: \n{y_test}\n")


if __name__ == "__main__":
    main()
