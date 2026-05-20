DATA_PATH = './data/clean_weibo_text.csv'
MODEL_DIR = './models/'
TFIDF_PARAMS = {
    'max_features' : 300000, 
    'analyzer' : 'word', 
    'ngram_range' : (1, 2)
}
GRID_PARAMS = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [4, 6, 8, 10],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2.0]
}
XGB_PARAMS = {
    'n_estimators': 400, 
    'max_depth' : 10, 
    'colsample_bytree' : 0.6, 
    'learning_rate' : 0.1, 
    'subsample' : 0.8, 
    'objective' : 'multi:softprob',
    'num_class' : 4, 
    'min_child_weight' : 1, 
    'reg_alpha' : 0, 
    'reg_lambda' : 1, 
    'random_state' : 16
}
LABELS = {
    0: '喜悦', 
    1: '愤怒', 
    2: '厌恶', 
    3: '低落'
}