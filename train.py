
#导入包
import config
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV


def load_data(path = config.DATA_PATH):

    #导入数据
    data = pd.read_csv(path)
    
    return data

def fit_tfidf_vectorizer(
        tests, 
        path = config.MODEL_DIR + 'tf_idf.pkl', 
        params = config.TFIDF_PARAMS
):

    #训练tf-idf
    #实例化Tf-IDF
    tf_idf = TfidfVectorizer(**params)

    #开始训练
    tf_idf.fit(tests)

    #保存到本地
    joblib.dump(value = tf_idf, filename = path)

    return

def train_model(
        XGB_params = config.XGB_PARAMS, 
        tf_idf_path = config.TFIDF_PARAMS, 
        model_path = config.MODEL_DIR + 'xgboost.pkl'
):
    #导入数据
    data = load_data()

    #分开特征和标签
    x_train = data['text']
    y_train = data['label']

    #导入训练好的tf_idf，将文本转化为向量
    tf_idf = joblib.load(tf_idf_path)

    #文本向量化
    x_train = tf_idf.transform(x_train)

    #实例化模型
    xgb_estimator = XGBClassifier(XGB_params)

    #开始训练模型
    #每训练50棵树就输出1次损失函数
    xgb_estimator.fit(
        x_train, 
        y_train, 
        eval_set = [(x_train, y_train)], 
        verbose = 50
    )

    print('训练完毕！')

    joblib.dump(value = xgb_estimator, filename = model_path)

    return

def GridSearchcv(
        estimator_path = config.MODEL_DIR + 'xgboost.pkl', 
        tf_idf_path = config.MODEL_DIR + 'tf_idf.pkl', 
        XGB_params = config.XGB_PARAMS, 
        grid_param = config.GRID_PARAMS, 
        estimator_have: bool = True
):
    #训练时间会很长，需要耐心等待
    data = load_data()

    #导入模型和tf_idf
    if estimator_have:
        xgb_estimator = joblib.load(filename = estimator_path)
    else:
        xgb_estimator = XGBClassifier(XGB_params)
    
    tf_idf = joblib.load(filename = tf_idf_path)

    #数据处理
    x_train = data['text']
    y_train = data['label']

    #将文本转化为向量
    x_train = tf_idf.transform(x_train)

    #开始交叉验证网格搜索
    xgb_estimator = GridSearchCV(
        estimator = xgb_estimator, 
        param_grid = grid_param, 
        cv = 3, 
        verbose = 2
    )

    xgb_estimator.fit(x_train, y_train)

    print('best_estimator_-->', xgb_estimator.best_estimator_)
    print('best_params_-->', xgb_estimator.best_params_)
    print('best_score_-->', xgb_estimator.best_score_)

    return



if __name__ == '__main__':
    print('测试模块')
    #GridSearchcv()
    


