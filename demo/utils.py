import shutil
import torch
from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def visualize_graph(model, writer, input_size=(1, 3, 32, 32)):
    dummy_input = torch.rand(input_size)
    # with SummaryWriter(comment=name) as w:
    writer.add_graph(model, (dummy_input, ))

def get_parameters_size(model):
    total = 0
    for p in model.parameters():
        _size = 1
        for i in range(len(p.size())):
            _size *= p.size(i)
        total += _size
    return total