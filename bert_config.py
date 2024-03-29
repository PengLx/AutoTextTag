import argparse
import os

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        parser.add_argument('--output_dir', default=os.path.join('./checkpoint'),
                            help='the output dir for model checkpoints')
        parser.add_argument('--bert_dir', default='./model/models--bert-base-cased/snapshots/5532cc56f74641d4bb33641f5c76a55d11f846e0', help = 'bert dir for uer')
        # other args
        parser.add_argument('--num_tags', default=59, type=int,
                            help='number of tags')  # 多标签分类的类别数
        parser.add_argument('--seed', type=int, default=123, help='random seed')
        parser.add_argument('--gpu_ids', type=str, default="0",
                            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')
        parser.add_argument('--max_seq_len', default=512, type=int)
        parser.add_argument('--batch_size', default=8, type=int) # 默认为 8
        parser.add_argument('--swa_start', default=2, type=int,
                            help='the epoch when swa start')
        parser.add_argument('--train_epochs', default=16, type=int,
                            help='Max training epoch')
        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')
        parser.add_argument('--lr', default=3e-5, type=float,
                            help='learning rate for the bert module')
        parser.add_argument('--other_lr', default=3e-4, type=float,
                            help='learning rate for the module except bert')
        parser.add_argument('--max_grad_norm', default=1, type=float,
                            help='max grad clip')
        parser.add_argument('--warmup_proportion', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--eval_model', default=True, action='store_true',
                            help='whether to eval model after training')
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()