import torch
import pickle
from sklearn.metrics import accuracy_score

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
    """
    Evaluates a model's top k accuracy

    Parameters:
        output (torch.autograd.Variable): model output
        target (torch.autograd.Variable): ground-truths/labels
        topk (list): list of integers specifying top-k precisions
            to be computed

    Returns:
        float: percentage of correct predictions
    """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def testset_Accuracy(task_model,test_dataloader,args):
    task_model.eval()
    total, correct = 0, 0
    if args.dataset == 'caltech':
        for imgs, labels in test_dataloader:
            if args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)# mu, std
            
            # print("labels sharpe",labels.shape)
            # print("preds sharpe",preds.shape)
            # if preds.size(0)>1:
            #     dummy=torch.Tensor(1,preds.size(1)*preds.size(0),args.num_classes)
            #     #print("preds sharpe",preds.shape,preds.size(1))
            #     preds=preds.view_as(dummy)
            print("preds sharpe",preds.shape)
            with open('preds', 'wb') as fp:pickle.dump(preds, fp)
            with open('labels', 'wb') as fp:pickle.dump(labels, fp)
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            # print("predictions shape is",preds.shape)
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100
    else:
        for imgs, labels in test_dataloader:
            if args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds, mu, std = task_model(imgs)
            if preds.size(0)>1:
                dummy=torch.Tensor(1,preds.size(1)*preds.size(0),args.num_classes)
                #print("preds sharpe",preds.shape,preds.size(1))
                preds=preds.view_as(dummy)
            # print("labels sharpe",labels.shape)
            # with open('preds', 'wb') as fp:pickle.dump(preds, fp)
            # with open('labels', 'wb') as fp:pickle.dump(labels, fp)
            preds = torch.argmax(preds[0], dim=1).cpu().numpy()
            # print("predictions shape is",preds.shape)
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100
        