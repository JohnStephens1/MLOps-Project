from text_classifier.data.data import get_dataset, get_train_test_df


def main():
    df = get_dataset()
    train, test = get_train_test_df(df)
    print(
        f'train: {train.head()}\ntest: {test.head()}'
    )


if __name__ == "__main__":
    main()
