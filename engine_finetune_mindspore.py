import os
import csv
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score
)
from pycm import ConfusionMatrix
import util_mindspore.misc as misc
import util_mindspore.lr_sched as lr_sched
from util_mindspore.datasets import Mixup


def train_one_epoch(
    model: nn.Cell,
    criterion: nn.Cell,
    data_loader: Iterable,
    optimizer: nn.Optimizer,
    device: str,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
):
    """Train the model for one epoch."""
    model.set_train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    
    # Define forward function for gradient computation
    def forward_fn(samples, targets):
        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)
        
        outputs = model(samples)
        loss = criterion(outputs, targets)
        return loss, outputs
    
    # Get gradient function
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        # Adjust learning rate
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples = Tensor(samples, ms.float32)
        targets = Tensor(targets, ms.int32)
        
        # Forward pass and compute gradients
        (loss, outputs), grads = grad_fn(samples, targets)
        
        # Apply gradients
        optimizer(grads)
        
        # Update metrics
        loss_value = loss.asnumpy().item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        metric_logger.update(loss=loss_value)
        
        lr = optimizer.get_lr().asnumpy().item() if hasattr(optimizer, 'get_lr') else args.lr
        metric_logger.update(lr=lr)
        
        if log_writer is not None:
            log_writer.add_scalar('train_loss', loss_value, epoch * len(data_loader) + data_iter_step)
            log_writer.add_scalar('lr', lr, epoch * len(data_loader) + data_iter_step)
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@ms.jit
def evaluate(data_loader, model, device, args):
    """Evaluate the model"""
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # Switch to evaluation mode
    model.set_train(False)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target = batch
        images = Tensor(images, ms.float32)
        target = Tensor(target, ms.int32)
        
        # Compute output
        output = model(images)
        loss = criterion(output, target)
        
        # Get predictions and probabilities
        probs = ops.softmax(output, axis=-1)
        preds = ops.argmax(output, axis=-1)
        
        all_preds.extend(preds.asnumpy().tolist())
        all_targets.extend(target.asnumpy().tolist())
        all_probs.extend(probs.asnumpy().tolist())
        
        metric_logger.update(loss=loss.asnumpy().item())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Accuracy
    acc = accuracy_score(all_targets, all_preds)
    
    # For multi-class classification
    if args.nb_classes > 2:
        # Macro and weighted averages
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        f1_weighted = f1_score(all_targets, all_preds, average='weighted')
        
        # AUC (one-vs-rest)
        try:
            auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
        except:
            auc = 0.0
        
        # Precision and Recall
        precision_macro = precision_score(all_targets, all_preds, average='macro')
        recall_macro = recall_score(all_targets, all_preds, average='macro')
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(all_targets, all_preds)
        
        print(f'* Acc@1 {acc:.3f} F1-macro {f1_macro:.3f} F1-weighted {f1_weighted:.3f}')
        print(f'* AUC {auc:.3f} Precision {precision_macro:.3f} Recall {recall_macro:.3f} Kappa {kappa:.3f}')
        
        # Confusion Matrix
        cm = ConfusionMatrix(actual_vector=all_targets, predict_vector=all_preds)
        print("Confusion Matrix:")
        print(cm)
        
        return {
            'acc1': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'auc': auc,
            'precision': precision_macro,
            'recall': recall_macro,
            'kappa': kappa,
            'loss': metric_logger.loss.global_avg
        }
    
    else:
        # Binary classification
        f1 = f1_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs[:, 1])
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        
        print(f'* Acc@1 {acc:.3f} F1 {f1:.3f} AUC {auc:.3f}')
        print(f'* Precision {precision:.3f} Recall {recall:.3f}')
        
        return {
            'acc1': acc,
            'f1': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'loss': metric_logger.loss.global_avg
        }


def save_results_to_csv(results, args, epoch=None):
    """Save evaluation results to CSV file"""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = os.path.join(output_dir, 'results.csv')
    
    # Prepare row data
    row_data = {
        'epoch': epoch if epoch is not None else 'final',
        'task': args.task,
        'model': args.model,
        'input_size': args.input_size,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        **results
    }
    
    # Write to CSV
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    
    print(f"Results saved to {csv_file}")


def plot_training_curves(train_losses, val_losses, val_accs, output_dir):
    """Plot training curves"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curve
    ax2.plot(epochs, val_accs, 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Cell):
    """Label smoothing cross entropy loss for MindSpore"""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def construct(self, x, target):
        logprobs = ops.log_softmax(x, axis=-1)
        nll_loss = -logprobs.gather(target.unsqueeze(1), 1)
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(axis=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]
    
    _, pred = ops.topk(output, maxk, axis=1)
    pred = pred.transpose()
    correct = ops.equal(pred, target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res 