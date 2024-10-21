import numpy as np
from sklearn.datasets import fetch_20newsgroups_vectorized
from tqdm import trange

# normalize表示是否对数据归一化，这里我们保留原始数据
# data_home是数据保存路径
train_data = fetch_20newsgroups_vectorized(subset='train',
    normalize=False, data_home='20newsgroups')
test_data = fetch_20newsgroups_vectorized(subset='test',
    normalize=False, data_home='20newsgroups')
print('文章主题：', '\n'.join(train_data.target_names))
print(train_data.data[0])

# 统计新闻主题频率
cat_cnt = np.bincount(train_data.target)
print('新闻数量：', cat_cnt)
log_cat_freq = np.log(cat_cnt / np.sum(cat_cnt))

# 对每个主题统计单词频率
alpha = 1.0
# 单词频率，20是主题个数，train_data.feature_names是分割出的单词
log_voc_freq = np.zeros((20, len(train_data.feature_names))) + alpha
# 单词计数，需要加上先验计数
voc_cnt = np.zeros((20, 1)) + len(train_data.feature_names) * alpha
# 用nonzero返回稀疏矩阵不为零的行列坐标
rows, cols = train_data.data.nonzero()
for i in trange(len(rows)):
    news = rows[i]
    voc = cols[i]
    cat = train_data.target[news] # 新闻类别
    log_voc_freq[cat, voc] += train_data.data[news, voc]
    voc_cnt[cat] += train_data.data[news, voc]

log_voc_freq = np.log(log_voc_freq / voc_cnt)


def test_news(news):
    rows, cols = news.nonzero()
    # 对数后验
    log_post = np.copy(log_cat_freq)
    for row, voc in zip(rows, cols):
        # 加上每个单词在类别下的后验
        log_post += log_voc_freq[:, voc]
    return np.argmax(log_post)




preds = []
for news in test_data.data:
    preds.append(test_news(news))
acc = np.mean(np.array(preds) == test_data.target)
print('分类准确率：', acc)


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=alpha)
mnb.fit(train_data.data, train_data.target)
print('分类准确率：', mnb.score(test_data.data, test_data.target))




