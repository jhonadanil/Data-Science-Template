TARGET = 'diagnosis'

SELECTED_FEATURES = [
    'perimeter_worst',
    'radius_worst',
    'area_worst',
    'concave points_mean',
    'concave points_worst',
    'radius_mean',
    'area_mean',
    'concavity_worst',
    'area_se',
    'concavity_mean',
]

BEST_PARAMS = {
    'class_weight': None,
    'max_iter': 400,
    'l1_ratio': 1,          # equivalent to penalty='l1' (deprecated in sklearn 1.8+)
    'solver': 'saga',
    'tol': 0.1,
}
