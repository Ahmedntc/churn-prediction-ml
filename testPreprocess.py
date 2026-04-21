from src.data.preprocess import loadData, prepareFeats, buildPreprocessor, splitData

df = loadData("data/processed/telco_clean.csv")
X, y = prepareFeats(df)
X_train, X_test, y_train, y_test = splitData(X, y)

preprocessor = buildPreprocessor()
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

print("X_train shape:", X_train_proc.shape)
print("X_test shape:", X_test_proc.shape)
print("y_train distribuição:\n", y_train.value_counts())
print("y_test distribuição:\n", y_test.value_counts())