import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Original Train Shape:", train.shape)
print("Original Test Shape:", test.shape)

print("\nTrain Missing Values:")
print(train.isnull().sum())

print("\nTrain Duplicate Rows:", train.duplicated().sum())

train = train.drop_duplicates()
test = test.drop_duplicates()


def process_features(df):
    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"])

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek

    df = pd.get_dummies(
        df,
        columns=["season", "weather"],
        dtype=int
    )

    return df


train = process_features(train)
test = process_features(test)


train = train.drop(
    ["casual", "registered", "datetime"],
    axis=1
)

test = test.drop(
    ["datetime"],
    axis=1
)


X_columns = train.drop("count", axis=1).columns

test = test.reindex(
    columns=X_columns,
    fill_value=0
)


train.to_csv("train_processed.csv", index=False)
test.to_csv("test_processed.csv", index=False)


print("\nProcessing Complete!")

print("Final Train Shape:", train.shape)
print("Final Test Shape:", test.shape)

print("\nFinal Train Columns:")
print(train.columns)

print("\nFinal Train Preview:")
print(train.head())