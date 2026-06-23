from sklearn.model_selection import train_test_split


def dividir_datos(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    Y = df[target]

    return train_test_split(
        X, Y,
        test_size=test_size,
        random_state=random_state
    )
