import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fasttext
import time
import os
#训练模型
# path = "fasttext"
# os.chdir(path)
# 监督训练模型
start = time.clock()
model = fasttext.train_supervised(
    input="train.txt",
    label_prefix="__label__" ,
    lr=0.05,
    epoch=25,
    # wordNgrams=2, # 该参数导致准确率降低
    bucket=200000,
    dim=50,
    loss="softmax"   # 可选loss='softmax'
    )
end = time.clock()
print('Running time: %s Seconds' % (end - start))
model.save_model("model_news_fasttext.bin")
#load训练好的模型
model = fasttext.load_model('model_news_fasttext.bin')
print('训练完成！')
