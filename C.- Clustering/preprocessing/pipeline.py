from sklearn.preprocessing import LabelEncoder


def codificar_genero(df):
    encoder = LabelEncoder()
    df['Gender'] = encoder.fit_transform(df['Gender'])
    return df, encoder
