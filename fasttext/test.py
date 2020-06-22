import fasttext
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

model = fasttext.load_model('model_news_fasttext.bin')
print(model.labels)

name=[ "ariculture","car","culture","edu","entertainment","finance","game","house","military","sports","stock",
       "story","tech","travel","world"]
labels_right = []
texts = []
# 获取测试集数据
with open("test.txt") as fr:
  for line in fr:
    line = str(line.encode("utf-8"), 'utf-8').rstrip()
    try:
      labels_right.append(line.split("\t")[1])
      texts.append(line.split("\t")[0])
    except:
      print(line)
labels_predict = [term[0] for term in model.predict(texts)[0]] #预测输出结果为二维形式
print(classification_report(labels_predict,labels_right))


