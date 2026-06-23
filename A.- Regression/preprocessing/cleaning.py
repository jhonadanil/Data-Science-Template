def limpieza(df):
    df = df.copy()

    df.rename(columns={
        'crim': 'Criminalidad',
        'zn': 'Terreno residencial',
        'indus': 'Comercio',
        'chas': 'Cerca del rio',
        'nox': 'Contaminación',
        'rm': 'Nº habitaciones',
        'age': 'Edad',
        'dis': 'Dis. centro empleo',
        'rad': 'Carretera Princ.',
        'tax': 'Impuestos',
        'ptratio': 'Prop. alumno/maestro',
        'b': 'Prop. población afroamericana',
        'lstat': 'Prop. bajos recursos',
        'medv': 'Precio',
    }, inplace=True)

    df.dropna(inplace=True)

    df["Prop. bajos recursos"] = -df["Prop. bajos recursos"]

    return df