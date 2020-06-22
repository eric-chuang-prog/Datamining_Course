import os
import math
import torch
import random
import json
from collections import Counter
import jieba

def read_lines(path):
    """
    {"label": "102",
    "label_desc": "news_entertainment",
    "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物",
    "keywords": "江疏影,美少女,经纪人,甜甜圈"}
    """
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            yield eval(line)
    f.close()

def read(data_path,list):
    for items in read_lines(data_path):
        sent = tuple(jieba.cut(items["sentence"], cut_all=False))
        label = items["label_desc"]
        example = [sent, label]
        list.append(example)
    return list

##生成fastext的训练和测试数据集

def write_texts(infile, outfile):
  print(infile)
  print(outfile)
  local_in_file = open(infile ,encoding="utf-8")
  local_out_file = open(outfile , "w", encoding="utf-8")
  lines=local_in_file.readlines()
  print(len(lines))

  local_lines=len(lines)
  count=0
  index=0

  for line in read_lines(infile):
    if(count%(local_lines/10)==0 and count>0):
      print(count)
      index+=1
    seg_list = jieba.cut(line["sentence"], cut_all=False)
    text=" ".join(seg_list)
    text1=text+"\t__label__"+line["label_desc"]+"\n"
    # text = line.replace("\n",str("\t__label__"+"\n"))
    local_out_file.write(text1)
    local_out_file.flush()
    count+=1
  print(count)
  local_in_file.close()
  local_out_file.close()

news_train="train.json"
news_test="test.json"
ftrain ="train.txt"
ftest ="test.txt"
write_texts(news_train, ftrain)
write_texts(news_test, ftest)
print('完成输出数据！')
