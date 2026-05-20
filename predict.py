
#导入包
import train
import config
import joblib
import jieba


def load_model(
        mode_path = config.MODEL_DIR + 'xgboost.pkl', 
        tf_idf_path = config.MODEL_DIR + 'tf_idf.pkl'
):
    
    estimator = joblib.load(mode_path)
    tf_idf = joblib.load(tf_idf_path)

    return estimator, tf_idf

def tokenize(text):

    return ' '.join(jieba.cut(text))

def predict(input):

    estimator, tf_idf = load_model()

    use_input = tf_idf.transform([tokenize(input)])

    predict_label = estimator.predict(use_input)[0]
    predict_proba = estimator.predict_proba(use_input)[0]

    ans = ''
    if predict_label == 0:
        ans = '喜悦'
    elif predict_label == 1:
        ans = '愤怒'
    elif predict_label == 2:
        ans = '厌恶'
    else:
        ans = '低落'

    print(f'输入: "{input}"')
    print(f' 预测情感类别: {ans}')
    print(f'各类别概率: 喜悦：{predict_proba[0]}，愤怒：{predict_proba[1]}，厌恶：{predict_proba[2]}，低落：{predict_proba[3]}')

if __name__ == '__main__':
    #测试用
    predict('我爱你')
    predict('切，你这有什么好伤心的')
    predict('今天天气真不错')
    predict('你这样做真的让我很愤怒')
    predict('唉，你也觉得我很恶心，我都懂')







