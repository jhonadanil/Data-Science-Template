from sklearn.preprocessing import StandardScaler, LabelEncoder

def estandarizar(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaler = scaler.fit_transform(x_train)
    x_test_scaler = scaler.transform(x_test)
    return x_train_scaler, x_test_scaler, scaler

def codificar (y_train, y_test):
    encoder = LabelEncoder()
    y_train_encoder = encoder.fit_transform(y_train)
    y_test_encoder = encoder.transform(y_test)
    return y_train_encoder, y_test_encoder, encoder