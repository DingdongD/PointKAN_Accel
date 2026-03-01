import argparse
import os
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from data import ModelNet40
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np
import math


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')  # 创建参数解析器
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')  # 检查点保存路径
    parser.add_argument('--msg', type=str, help='message after checkpoint')  # 检查点后附加消息
    parser.add_argument('--batch_size', type=int, default=18, help='batch size in training')  # 训练批量大小
    parser.add_argument('--model', default='pointKAN', help='model name [default: pointnet_cls]')  # 模型名称
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')  # 训练轮数
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')  # 点云点数
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')  # 学习率
    parser.add_argument('--min_lr', default=0.005, type=float, help='min lr')  # 最小学习率
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='decay rate')  # 权重衰减
    parser.add_argument('--seed', type=int, default=42, help='random seed')  # 随机种子，默认值为 42
    parser.add_argument('--workers', default=8, type=int, help='workers')  # 数据加载 worker 数量
    parser.add_argument('--resume', type=str, default=None,
                        help='断点重训：指定检查点目录或 .pth 文件路径，从该处恢复训练')  # 断点恢复路径
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值：验证损失连续多少 epoch 不下降则停止，0 表示不早停')  # 过拟合保护
    return parser.parse_args()  # 解析并返回参数


def main():
    args = parse_args()  # 解析命令行参数
    if args.seed is None:  # 如果没有指定随机种子
        args.seed = np.random.randint(1, 100)  # 生成一个 1-99 之间的随机种子
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # 禁用 HDF5 文件锁定

    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."  # 确保在 CUDA 环境中运行
    device = 'cuda'  # 设置设备为 CUDA
    if args.seed is not None:  # 如果设置了随机种子
        torch.manual_seed(args.seed)  # 设置 PyTorch 随机种子
        np.random.seed(args.seed)  # 设置 NumPy 随机种子
        torch.cuda.manual_seed_all(args.seed)  # 设置所有 CUDA 设备的随机种子
        torch.cuda.manual_seed(args.seed)  # 设置当前 CUDA 设备的随机种子
        torch.set_printoptions(10)  # 设置打印选项
        torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 基准测试
        torch.backends.cudnn.deterministic = True  # 启用 CuDNN 确定性模式
        os.environ['PYTHONHASHSEED'] = str(args.seed)  # 设置 Python 哈希种子
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))  # 生成时间字符串
    if args.msg is None:  # 如果没有指定消息
        message = time_str  # 使用时间字符串作为消息
    else:
        message = "-" + args.msg  # 使用指定的消息
    # 断点重训：--resume 指定目录或 .pth 文件时，从该路径恢复并继续保存到同目录
    resume_checkpoint_file = None  # 要加载的检查点文件（仅当 --resume 时非 None）
    if args.resume is not None:
        r = args.resume.rstrip('/')
        if r.endswith('.pth'):
            resume_checkpoint_file = r
            args.checkpoint = os.path.dirname(r)
        else:
            args.checkpoint = r
            resume_checkpoint_file = os.path.join(args.checkpoint, "last_checkpoint.pth")
        if not os.path.isfile(resume_checkpoint_file):
            raise FileNotFoundError(f"断点重训失败：未找到检查点文件 {resume_checkpoint_file}")
    else:
        args.checkpoint = 'checkpoints/' + args.model + message + '-' + str(args.seed)  # 构建检查点路径
    if not os.path.isdir(args.checkpoint):  # 如果检查点目录不存在
        mkdir_p(args.checkpoint)  # 创建检查点目录

    screen_logger = logging.getLogger("Model")  # 创建日志记录器
    screen_logger.setLevel(logging.INFO)  # 设置日志级别为 INFO
    formatter = logging.Formatter('%(message)s')  # 创建日志格式化器
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))  # 创建文件处理器
    file_handler.setLevel(logging.INFO)  # 设置文件处理器级别
    file_handler.setFormatter(formatter)  # 设置文件处理器格式化器
    screen_logger.addHandler(file_handler)  # 添加文件处理器到日志记录器

    def printf(str):  # 定义打印函数
        screen_logger.info(str)  # 记录日志
        print(str)  # 打印到控制台

    # Model
    printf(f"args: {args}")  # 打印参数
    printf('==> Building model..')  # 打印构建模型信息
    net = models.__dict__[args.model]()  # 根据模型名称创建模型实例
    criterion = cal_loss  # 设置损失函数
    net = net.to(device)  # 将模型移至 CUDA 设备
    # criterion = criterion.to(device)  # 注释掉的代码，损失函数通常不需要移至设备
    if device == 'cuda':  # 如果使用 CUDA
        net = torch.nn.DataParallel(net)  # 使用数据并行
        cudnn.benchmark = True  # 启用 CuDNN 基准测试

    best_test_acc = 0.  # 最佳测试准确率
    best_train_acc = 0.  # 最佳训练准确率
    best_test_acc_avg = 0.  # 最佳测试平均准确率
    best_train_acc_avg = 0.  # 最佳训练平均准确率
    best_test_loss = float("inf")  # 最佳测试损失
    best_train_loss = float("inf")  # 最佳训练损失
    best_epoch = -1  # 最佳模型所在 epoch（用于早停后汇报）
    start_epoch = 0  # 开始轮数
    optimizer_dict = None  # 优化器状态字典
    scheduler_dict = None  # 学习率调度器状态（断点恢复用）

    load_path = resume_checkpoint_file if resume_checkpoint_file is not None else os.path.join(args.checkpoint, "last_checkpoint.pth")
    if not os.path.isfile(load_path):  # 如果没有最后检查点
        save_args(args)  # 保存参数
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)  # 创建日志记录器
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-acc-B', 'Train-acc',
                          'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])  # 设置日志名称
    else:  # 断点重训：从检查点恢复
        printf(f"Resuming from checkpoint: {load_path}")  # 打印恢复检查点信息
        checkpoint = torch.load(load_path)  # 加载检查点
        net.load_state_dict(checkpoint['net'])  # 加载模型状态
        start_epoch = checkpoint['epoch']  # 加载开始轮数
        best_test_acc = checkpoint['best_test_acc']  # 加载最佳测试准确率
        best_train_acc = checkpoint['best_train_acc']  # 加载最佳训练准确率
        best_test_acc_avg = checkpoint['best_test_acc_avg']  # 加载最佳测试平均准确率
        best_train_acc_avg = checkpoint['best_train_acc_avg']  # 加载最佳训练平均准确率
        best_test_loss = checkpoint['best_test_loss']  # 加载最佳测试损失
        best_train_loss = checkpoint['best_train_loss']  # 加载最佳训练损失
        best_epoch = checkpoint.get('best_epoch', start_epoch - 1)  # 最佳 epoch
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model, resume=True)  # 恢复日志记录器
        optimizer_dict = checkpoint['optimizer']  # 加载优化器状态
        scheduler_dict = checkpoint.get('scheduler')  # 学习率调度器状态（可选，用于完整恢复）

    printf('==> Preparing data..')  # 打印准备数据信息
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)  # 创建训练数据加载器
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)  # 创建测试数据加载器

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)  # 创建 SGD 优化器
    if optimizer_dict is not None:  # 如果有优化器状态
        optimizer.load_state_dict(optimizer_dict)  # 加载优化器状态
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr, last_epoch=start_epoch - 1)  # 创建余弦退火学习率调度器
    if scheduler_dict is not None:  # 断点恢复时恢复调度器状态（学习率曲线一致）
        scheduler.load_state_dict(scheduler_dict)

    epochs_no_improve = 0  # 验证损失连续未改善的 epoch 数（用于早停）
    for epoch in range(start_epoch, args.epoch):  # 训练循环
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))  # 打印当前轮数和学习率
        train_out = train(net, train_loader, optimizer, criterion, device)  # 训练模型
        test_out = validate(net, test_loader, criterion, device)  # 验证模型
        scheduler.step()  # 更新学习率

        if test_out["acc"] > best_test_acc:  # 如果当前测试准确率高于最佳测试准确率
            best_test_acc = test_out["acc"]  # 更新最佳测试准确率
            is_best = True  # 设置为最佳模型
            best_epoch = epoch  # 记录最佳 epoch
        else:
            is_best = False  # 不是最佳模型

        # 早停：按验证损失判断是否改善
        if test_out["loss"] < best_test_loss:
            best_test_loss = test_out["loss"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc  # 更新最佳测试准确率
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc  # 更新最佳训练准确率
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg  # 更新最佳测试平均准确率
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg  # 更新最佳训练平均准确率
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss  # 更新最佳训练损失

        save_model(
            net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,  # 保存最佳测试准确率
            best_train_acc=best_train_acc,  # 保存最佳训练准确率
            best_test_acc_avg=best_test_acc_avg,  # 保存最佳测试平均准确率
            best_train_acc_avg=best_train_acc_avg,  # 保存最佳训练平均准确率
            best_test_loss=best_test_loss,  # 保存最佳测试损失
            best_train_loss=best_train_loss,  # 保存最佳训练损失
            best_epoch=best_epoch,  # 保存最佳 epoch（早停后汇报用）
            optimizer=optimizer.state_dict(),  # 保存优化器状态
            scheduler=scheduler.state_dict()  # 保存学习率调度器状态（断点重训用）
        )
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc_avg"], train_out["acc"],
                       test_out["loss"], test_out["acc_avg"], test_out["acc"]])  # 记录日志
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")  # 打印训练结果
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")  # 打印测试结果

        # 早停：验证损失连续 patience 个 epoch 未下降则停止
        if args.patience > 0 and epochs_no_improve >= args.patience:
            printf(f"[Early Stopping] 验证损失已连续 {args.patience} 个 epoch 未改善，停止训练。最佳模型已保存于 best_checkpoint.pth (epoch {best_epoch + 1})")
            break
    logger.close()  # 关闭日志记录器

    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)  # 打印最终结果标题
    printf(f"++  Best epoch: {best_epoch + 1} (best_checkpoint.pth)  ++")  # 最佳模型所在 epoch
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")  # 打印最后训练和测试时间
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")  # 打印最佳训练和测试损失
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")  # 打印最佳训练和测试平均准确率
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")  # 打印最佳训练和测试准确率
    printf(f"++++++++" * 5)  # 打印结束标记

def train(net, trainloader, optimizer, criterion, device):
    net.train()  # 设置模型为训练模式
    train_loss = 0  # 训练损失
    correct = 0  # 正确预测数
    total = 0  # 总样本数
    train_pred = []  # 训练预测结果
    train_true = []  # 训练真实标签
    time_cost = datetime.datetime.now()  # 开始时间
    for batch_idx, (data, label) in enumerate(trainloader):  # 遍历训练数据
        data, label = data.to(device), label.to(device).squeeze()  # 将数据和标签移至设备并挤压维度
        data = data.permute(0, 2, 1)  # 调整数据维度，变为 [batch, 3, 1024]
        optimizer.zero_grad()  # 清零梯度
        logits = net(data)  # 前向传播
        loss = criterion(logits, label)  # 计算损失
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()  # 更新参数
        train_loss += loss.item()  # 累加损失
        preds = logits.max(dim=1)[1]  # 获取预测结果

        train_true.append(label.cpu().numpy())  # 保存真实标签
        train_pred.append(preds.detach().cpu().numpy())  # 保存预测结果

        total += label.size(0)  # 累加总样本数
        correct += preds.eq(label).sum().item()  # 累加正确预测数

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))  # 显示进度条

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())  # 计算训练时间
    train_true = np.concatenate(train_true)  # 合并真实标签
    train_pred = np.concatenate(train_pred)  # 合并预测结果
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),  # 平均损失
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),  # 准确率
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),  # 平均准确率
        "time": time_cost  # 训练时间
    }


def validate(net, testloader, criterion, device):
    net.eval()  # 设置模型为评估模式
    test_loss = 0  # 测试损失
    correct = 0  # 正确预测数
    total = 0  # 总样本数
    test_true = []  # 测试真实标签
    test_pred = []  # 测试预测结果
    time_cost = datetime.datetime.now()  # 开始时间
    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (data, label) in enumerate(testloader):  # 遍历测试数据
            data, label = data.to(device), label.to(device).squeeze()  # 将数据和标签移至设备并挤压维度
            data = data.permute(0, 2, 1)  # 调整数据维度
            logits = net(data)  # 前向传播
            loss = criterion(logits, label)  # 计算损失
            test_loss += loss.item()  # 累加损失
            preds = logits.max(dim=1)[1]  # 获取预测结果
            test_true.append(label.cpu().numpy())  # 保存真实标签
            test_pred.append(preds.detach().cpu().numpy())  # 保存预测结果
            total += label.size(0)  # 累加总样本数
            correct += preds.eq(label).sum().item()  # 累加正确预测数
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))  # 显示进度条

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())  # 计算测试时间
    test_true = np.concatenate(test_true)  # 合并真实标签
    test_pred = np.concatenate(test_pred)  # 合并预测结果
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),  # 平均损失
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),  # 准确率
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),  # 平均准确率
        "time": time_cost  # 测试时间
    }

if __name__ == '__main__':
    main()
