import os
import csv
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score
)
from pycm import ConfusionMatrix
import util.misc as misc
import util.lr_sched as lr_sched


class Mixup:
    """简单的Mixup实现"""
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=0.0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def __call__(self, x, target):
        if np.random.rand() > self.prob:
            return x, target
            
        if self.mixup_alpha > 0 and np.random.rand() < self.switch_prob:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = x.shape[0]
            rand_index = paddle.randperm(batch_size)
            
            mixed_x = lam * x + (1 - lam) * x[rand_index]
            target_a, target_b = target, target[rand_index]
            return mixed_x, (target_a, target_b, lam)
        else:
            return x, target


def train_one_epoch(
    model: paddle.nn.Layer,
    criterion: paddle.nn.Layer,
    data_loader: Iterable,
    optimizer: paddle.optimizer.Optimizer,
    device: str,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
):
    """Train the model for one epoch."""
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.clear_grad()
    
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples = paddle.to_tensor(samples) if not isinstance(samples, paddle.Tensor) else samples
        targets = paddle.to_tensor(targets) if not isinstance(targets, paddle.Tensor) else targets
        
        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)
        
        with paddle.amp.auto_cast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        loss_value = loss.item()
        loss /= accum_iter
        
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.clear_grad()
        
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        if hasattr(optimizer, '_learning_rate'):
            lr = optimizer._learning_rate if hasattr(optimizer._learning_rate, 'get_lr') else optimizer._learning_rate
            if hasattr(lr, 'get_lr'):
                max_lr = lr.get_lr()
                min_lr = lr.get_lr()
            else:
                max_lr = lr
                min_lr = lr
        else:
            max_lr = optimizer.get_lr()
            min_lr = optimizer.get_lr()

        metric_logger.update(lr=max_lr)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    """Evaluate the model."""
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    
    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []
    
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images = paddle.to_tensor(batch[0]) if not isinstance(batch[0], paddle.Tensor) else batch[0]
        target = paddle.to_tensor(batch[-1]) if not isinstance(batch[-1], paddle.Tensor) else batch[-1]
        target_onehot = F.one_hot(target.cast('int64'), num_classes=num_class)
        
        with paddle.amp.auto_cast():
            output = model(images)
            loss = criterion(output, target)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        pred_softmax_value = F.softmax(output, axis=-1).cpu().numpy()
        pred_onehot_value = F.one_hot(paddle.argmax(output, axis=-1), num_classes=num_class).cpu().numpy()
        
        true_onehot.extend(target_onehot.cpu().numpy())
        pred_onehot.extend(pred_onehot_value)
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(paddle.argmax(output, axis=-1).cpu().numpy())
        pred_softmax.extend(pred_softmax_value)
    
    # 计算指标
    true_onehot = np.array(true_onehot)
    pred_onehot = np.array(pred_onehot)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_softmax = np.array(pred_softmax)
    
    # 计算各种指标
    accuracy_value = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1_micro = f1_score(true_labels, pred_labels, average='micro', zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    try:
        auc_macro = roc_auc_score(true_onehot, pred_softmax, average='macro', multi_class='ovr')
        auc_weighted = roc_auc_score(true_onehot, pred_softmax, average='weighted', multi_class='ovr')
    except:
        auc_macro = 0.0
        auc_weighted = 0.0
    
    precision_macro = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    
    # 写入CSV文件
    csv_file = os.path.join(args.output_dir, args.task, f'{mode}_results.csv')
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Epoch', 'Loss', 'Accuracy', 'F1_Macro', 'F1_Micro', 'F1_Weighted', 
                           'AUC_Macro', 'AUC_Weighted', 'Precision_Macro', 'Recall_Macro'])
        writer.writerow([epoch, metric_logger.meters['loss'].global_avg, accuracy_value, 
                        f1_macro, f1_micro, f1_weighted, auc_macro, auc_weighted, 
                        precision_macro, recall_macro])
    
    print(f'{mode} Results - Accuracy: {accuracy_value:.4f}, F1-Macro: {f1_macro:.4f}, AUC-Macro: {auc_macro:.4f}')
    
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def accuracy(output, target, topk=(1,)):
    """计算准确率"""
    with paddle.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.equal(target.reshape([1, -1]).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape([-1]).cast('float32').sum(0, keepdim=True)
            res.append(correct_k.multiply(paddle.to_tensor(100.0 / batch_size)))
        return res
