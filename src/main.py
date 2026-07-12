from data.data import *


def main():
    df = get_dataset()
    train, test = get_train_test_df(df)
    print(
        f'train: {train.head()}\ntest: {test.head()}'
    )


if __name__ == "__main__":
    main()
