# -*- coding: utf-8 -*-
import os
import torch
from config import parse_config
from data_loader import DataBatchIterator
from data_loader import PAD
from train import build_textcnn_model
from train import  train_textcnn_model
from torch.nn import CrossEntropyLoss
from torch.nn import Softmax
from sklearn.metrics import classification_report
def main():
    # 读配置文件
    config = parse_config()
    # 载入训练集合
    train_data = DataBatchIterator(
        config=config,
        is_train=True,
        dataset="train",
        batch_size=config.batch_size,
        shuffle=True)
    train_data.load()

    vocab = train_data.vocab

    # 载入测试集合
    valid_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="test",
        batch_size=config.batch_size)
    valid_data.set_vocab(vocab)
    valid_data.load()

    # 测试时
    checkpoint = torch.load(config.save_model+".pt",
                         map_location = config.device)
    model=checkpoint
    # model = build_textcnn_model(
    #     vocab, config, train=True)
    model.eval()
    total_loss = 0
    valid_data_iter = iter(valid_data)
    total_predict=torch.LongTensor ([])
    total_label=torch.LongTensor ([])
    for idx, batch in enumerate(valid_data_iter):
        model.zero_grad()
        ground_truth = batch.label
        # batch_first = False
        outputs = model(batch.sent)
        predict=torch.max(outputs,1)[1]
        total_predict=torch.cat((total_predict,predict),0)
        total_label =torch.cat((total_label,batch.label),0)
    print(classification_report(total_label,total_predict))

if __name__ == "__main__":
    main()
