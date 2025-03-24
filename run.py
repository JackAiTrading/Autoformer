import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer 系列用于时间序列预测')

    # 基本配置
    parser.add_argument('--is_training', type=int, required=True, default=1, help='状态（是否进行训练）')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='模型标识')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='模型名称，选项：[Autoformer, Informer, Transformer]')

    # 数据加载器
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='数据集类型')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='数据文件的根路径')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件')
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务，选项：[M, S, MS]; M:多变量预测多变量, S:单变量预测单变量, MS:多变量预测单变量')
    parser.add_argument('--target', type=str, default='OT', help='在 S 或 MS 任务中的目标特征')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间特征编码的频率，选项：[s:秒级, t:分钟级, h:小时级, d:日级, b:工作日, w:周级, m:月级]，也可以使用更详细的频率如 15min 或 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点的位置')

    # 预测任务
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=48, help='起始标记长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')

    # 模型定义
    parser.add_argument('--bucket_size', type=int, default=4, help='适用于 Reformer 的桶大小')
    parser.add_argument('--n_hashes', type=int, default=4, help='适用于 Reformer 的哈希数量')
    parser.add_argument('--enc_in', type=int, default=7, help='编码器输入大小')
    parser.add_argument('--dec_in', type=int, default=7, help='解码器输入大小')
    parser.add_argument('--c_out', type=int, default=7, help='输出大小')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='头的数量')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='全连接层维度')
    parser.add_argument('--moving_avg', type=int, default=25, help='移动平均窗口大小')
    parser.add_argument('--factor', type=int, default=1, help='注意力因子')
    parser.add_argument('--distil', action='store_false',
                        help='是否在编码器中使用蒸馏，使用此参数表示不使用蒸馏',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='丢弃率')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码，选项：[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    parser.add_argument('--output_attention', action='store_true', help='是否在编码器中输出注意力')
    parser.add_argument('--do_predict', action='store_true', help='是否预测未见过的未来数据')

    # 优化
    parser.add_argument('--num_workers', type=int, default=10, help='数据加载器的工作线程数')
    parser.add_argument('--itr', type=int, default=2, help='实验次数')
    parser.add_argument('--train_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='训练输入数据的批量大小')
    parser.add_argument('--patience', type=int, default=3, help='提前停止的耐心值')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='优化器学习率')
    parser.add_argument('--des', type=str, default='test', help='实验描述')
    parser.add_argument('--loss', type=str, default='mse', help='损失函数')
    parser.add_argument('--lradj', type=str, default='type1', help='调整学习率')
    parser.add_argument('--use_amp', action='store_true', help='是否使用自动混合精度训练', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用 gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu 编号')
    parser.add_argument('--use_multi_gpu', action='store_true', help='是否使用多个 gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='多 gpu 的设备编号')


    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
