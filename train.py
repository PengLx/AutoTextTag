import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import preprocess
import dataset
import models
import bert_config

args = bert_config.Args().get_parser()
label2id = {}
id2label = {}
labels = ['Exercise & Fitnesss', 'Strategy', 'Product & Design', 'Parenting', 'Music', 'Computer Science', 'longevity', 'History', 'Fashion & Beauty', 'Science Fiction', 'Entrepreneurship', 'Reading & Writing', 'Time Management', 'Cryptocurrency', 'Love & Relationships', 'Learning & Education', 'Motivation & Inspiration', 'Startups', 'Human Resources', 'Food', 'Politics', 'Arts & Culture', 'Philosophy', 'Marketing & Sales', 'Productivity', 'Meditation', 'Remote Work', 'Health', 'Cybersecurity', 'Movies & Shows', 'Videos', 'Money & Investments', 'Artificial Intelligence', 'Career', 'Mental Health', 'Business', 'space', 'Creativity', 'Science & Nature', 'Podcasts', 'anime', 'Entertainment', 'Travel', 'Leadership & Management', 'Society', 'Personal Development', 'Problem Solving', 'Psychology', 'Technology & The Future', 'Communication', 'Teamwork', 'Corporate Culture', 'Economics', 'Books', 'Religion & Spirituality', 'Habits', 'Sports', 'Mindfulness', 'softwareengineering']

for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader):
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.model = models.BertMLClf(args)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model.to(self.device)


    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss


    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)


    def train(self):
        total_step = len(self.train_loader) * self.args.train_epochs
        global_step = 0
        eval_step = 1000
        best_dev_micro_f1 = 0.0
        for epoch in range(args.train_epochs):
            for train_step, train_data in enumerate(self.train_loader):
                self.model.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                labels = train_data['labels'].to(self.device)
                train_outputs = self.model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(train_outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(
                    "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss.item()))
                global_step += 1
                if global_step % eval_step == 0:
                    dev_loss, dev_outputs, dev_targets = self.dev()
                    accuracy, micro_f1, macro_f1 = self.get_metrics(dev_outputs, dev_targets)
                    print(
                        "【dev】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(dev_loss, accuracy,
                                                                                                   micro_f1, macro_f1))
                    if macro_f1 > best_dev_micro_f1:
                        print("------------>save the best")
                        checkpoint = {
                            'epoch': epoch,
                            'loss': dev_loss,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        best_dev_micro_f1 = macro_f1
                        checkpoint_path = os.path.join(self.args.output_dir, 'best.pt')
                        self.save_ckp(checkpoint, checkpoint_path)


    def dev(self):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.dev_loader):
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                labels = dev_data['labels'].to(self.device)
                outputs = self.model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                outputs = (np.array(outputs) > 0.5).astype(int)
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(labels.cpu().detach().numpy().tolist())
        return total_loss, dev_outputs, dev_targets


    def test(self, checkpoint_path):
        model = self.model
        optimizer = self.optimizer
        model, optimizer, epoch, loss = self.load_ckp(model, optimizer, checkpoint_path)
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_loader):
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                labels = test_data['labels'].to(self.device)
                outputs = model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                outputs = (np.array(outputs) > 0.6).astype(int)
                test_outputs.extend(outputs.tolist())
                test_targets.extend(labels.cpu().detach().numpy().tolist())
        return total_loss, test_outputs, test_targets


    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, micro_f1, macro_f1



    def get_classification_report(self, outputs, targets, labels):
        report = classification_report(targets, outputs, target_names=labels)
        return report



    def output_to_label(self, output):
        probs = {}
        results = []
        for attr, prob in zip(id2label.values(), output):
            probs[attr] = prob
            if prob > 0.5:
                results.append([attr, prob])
        if len(results) != 0:
            return results, probs
        else:  # 没有超过阈值的则一个标签都无法命中
            return [], probs



    def batch_predict(self, model, infer_list, batch_size=16):
        features, callback_info = preprocess.out("infer_file", args, label2id, 'infer', infer_list)
        infer_dataset = dataset.MLDataset(features)
        infer_loader = DataLoader(dataset=infer_dataset,
                                  batch_size=batch_size,
                                  num_workers=2, shuffle=False)
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(infer_loader):
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                outputs = model(token_ids, attention_masks, token_type_ids)
                outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                test_outputs.extend(outputs)

        batch_results = []
        batch_probs = []
        for sample in test_outputs:
            result, prob = self.output_to_label(sample)
            batch_results.append(result)
            batch_probs.append(prob)
        return batch_results, batch_probs


if __name__ == '__main__':
    train_out = preprocess.out('./data/train_lighthouse.csv', args, label2id, 'train')
    features, callback_info = train_out
    train_dataset = dataset.MLDataset(features)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=2)
    dev_out = preprocess.out('./data/test_lighthouse.csv', args, label2id, 'dev')
    dev_features, dev_callback_info = dev_out
    dev_dataset = dataset.MLDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            num_workers=2)
    trainer = Trainer(args, train_loader, dev_loader, dev_loader)  # 测试集此处同dev
    # 训练和验证
    trainer.train()
    # 测试
    print('========进行测试========')
    checkpoint_path = './checkpoint/best.pt'
    total_loss, test_outputs, test_targets = trainer.test(checkpoint_path)
    accuracy, micro_f1, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
    print("【test】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(total_loss, accuracy, micro_f1,
                                                                                      macro_f1))
    report = trainer.get_classification_report(test_outputs, test_targets, labels)
    print(report)
