from src.data.preprocess import load_data, prepare_feats, build_preprocessor, split_data

df = load_data("dataframe/processed/telco_clean.csv")
X, y = prepare_feats(df)
X_train, X_test, y_train, y_test = split_data(X, y)

preprocessor = build_preprocessor()
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

print("X_train shape:", X_train_proc.shape)
print("X_test shape:", X_test_proc.shape)
print("y_train distribuição:\n", y_train.value_counts())
print("y_test distribuição:\n", y_test.value_counts())