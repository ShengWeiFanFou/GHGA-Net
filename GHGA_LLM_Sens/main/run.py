import argparse
import os

import paddle
import time

from trainer import set_seed, Trainer

if __name__ == '__main__':
    os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
    os.environ["CUDA_VISIBLE_DEVICES"]='gpu:0'
    start = time.time()
    parser = argparse.ArgumentParser(description = "model params")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", "-d", type=str, default='ours',help='thucnews tnews iflytek csldcp')
    parser.add_argument("--train_num", type=int, default=200)
    parser.add_argument("--file_dir", "-f_dir", type=str, default='./')
    parser.add_argument("--data_path", "-d_path", type=str, default='../data_processed/')
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--drop_out", type=float, default=0.7)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument('--batch_size',type=int,default=500)
    parser.add_argument('--trained',type=int,default=0)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--do_eval',type=bool,default=0)
    parser.add_argument('--predict', type=bool, default=0)
    params = parser.parse_args()

    if not params.disable_cuda and paddle.device.get_device():
        params.device = paddle.device.set_device('gpu:%d' % params.gpu)
    else:
        params.device = paddle.device.set_device('cpu')

    set_seed(params.seed)
    trainer = Trainer(params)
    do_eval=params.do_eval
    if do_eval:
         test_acc,best_f1 = trainer.eval()
    else:
        test_acc,best_f1 = trainer.train()
    del trainer
    print('total time: ', time.time() - start)