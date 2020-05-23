import time
import torch
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy
import numpy as np
from lib.Utility.metrics import testset_Accuracy
from sklearn.metrics import accuracy_score
import pickle
import torch.nn as nn
import copy
import torch.optim as optim

def plot_visdom(vis,x,y,winName,plotName):
    options = dict(fillarea=False,width=400,height=400,xlabel='Iteration',ylabel='Loss',title=winName)
    if (vis.win_exists(winName)==False):
        win = vis.line(X=np.array([0]),Y=np.array([0]),win=winName,name=plotName,opts=options)
    else:
        vis.line(X=np.array([x]),Y=np.array([y]),win=winName,update='append',name=plotName)

def read_data(dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

def train(Dataset,validate,test_dataloader, task_model, criterion, epoch, visdom, args,split,iterations):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int), denoising_noise_value (float) and var_beta (float).
    """
    #print("came to regular train")

    # Create instances to accumulate losses etc.
    optim_task_model = optim.SGD(task_model.parameters(), lr=0.001, momentum=0.9)
    task_model.train()
    if args.cuda:
        print("came to cud")
        task_model = task_model.cuda()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()

    # switch to train mode
    
    labeled_data = read_data(Dataset)
    
    end = time.time()
    acc=0
    # train
    #model.train()
    for iter_count in range(0,15000):
        inp, target = next(labeled_data)
    #for i, (inp, target,_) in enumerate(Dataset):
        if args.cuda:
            inp = inp.cuda()
            target = target.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute model forward
        output = task_model(inp)

        # calculate loss
        loss =  criterion(output, target)

        # record precision/accuracy and losses
        prec1 = accuracy(output, target)[0]
        top1.update(prec1.item(), inp.size(0))
        losses.update(loss, inp.size(0))

        # compute gradient and do SGD step
        optim_task_model.zero_grad()
        loss.backward()
        optim_task_model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if iter_count % args.print_freq == 0:
            print('Training: [{0}][{1}/{2}]\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch+1, iter_count, len(Dataset), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            plot_visdom(visdom,iterations,loss.item(),str(int(split*100))+'_loss','loss')
            best_model = copy.deepcopy(task_model)
            best_model = best_model.cuda()
            acc=testset_Accuracy(best_model,test_dataloader,args)
            plot_visdom(visdom,iterations,acc,str(int(split*100))+'_acc','acc')
        iterations=iterations+1
    # # TensorBoard summary logging
    # writer.add_scalar('training/train_precision@1', top1.avg, epoch)
    # writer.add_scalar('training/train_class_loss', losses.avg, epoch)
    # writer.add_scalar('training/train_average_loss', losses.avg, epoch)

    print(' * Train: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))


    return iterations,acc,loss


def train_var(Dataset,validate,test_dataloader, model, criterion, epoch, visdom, args,split,iterations):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int), denoising_noise_value (float) and var_beta (float).
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    labeled_data = read_data(Dataset)
    if args.cuda:
        print("came to cud")
        model = model.cuda()
    # Create instances to accumulate losses etc.
    cl_losses = AverageMeter()
    kld_losses = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    acc=0
    best_acc=0
    # train
    lr_change = [int(len(Dataset)*200),int(len(Dataset)*250),int(len(Dataset)*300)]#self.args.train_iterations // 4
    total=int((args.epochs*args.batch_size*len(Dataset))/128)
    print("Total Iterations are",total)
    print("LR Changes are",lr_change)
    for iter_count in range(0,total):
        if (iter_count in lr_change):
            print("Came for learning rate change",iter_count)
            for param in optimizer.param_groups:
                param['lr'] = param['lr'] / 10
        inp, target = next(labeled_data)
        if args.cuda:
            inp = inp.cuda()
            target = target.cuda()
        # measure data loading time
        data_time.update(time.time() - end)

        # compute model forward
        output_samples, mu, std = model(inp)
        if output_samples.size(0)>1:
            dummy=torch.Tensor(1,output_samples.size(0)*output_samples.size(1),args.num_classes)
            #print("classamples size", output_samples.size(0), output_samples.shape)
            output_samples=output_samples.view_as(dummy)
        # calculate loss
        # with open('output_samples_', 'wb') as fp:pickle.dump(output_samples, fp)
        # with open('target', 'wb') as fp:pickle.dump(target, fp)
        cl_loss, kld_loss = criterion(output_samples, target, mu, std, args.device)

        # add the individual loss components together and weight the KL term.
        #print("only classification")
        loss = cl_loss + args.var_beta * kld_loss

        # take mean to compute accuracy. Note if variational samples are 1 this only gets rid of a dummy dimension.
        output = torch.mean(output_samples, dim=0)

        # record precision/accuracy and losses
        prec1 = accuracy(output, target)[0]
        top1.update(prec1.item(), inp.size(0))

        losses.update((cl_loss + kld_loss).item(), inp.size(0))
        cl_losses.update(cl_loss.item(), inp.size(0))
        kld_losses.update(kld_loss.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print progress
        if iter_count % args.print_freq == 0:
            print('Training: [{0}][{1}/{2}]\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                   epoch+1, iter_count, len(Dataset), batch_time=batch_time,
                   data_time=data_time, loss=losses, cl_loss=cl_losses, top1=top1, KLD_loss=kld_losses))
            plot_visdom(visdom,iter_count,loss.item(),str(int(split*100))+'_loss','loss')
            acc=testset_Accuracy(model,test_dataloader,args)
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(model)
                best_optimum=optimizer.state_dict()
            #acc, loss = validate(test_dataloader, model, criterion, epoch, visdom, args.device, args)
            plot_visdom(visdom,iter_count,acc,str(int(split*100))+'_acc','acc')
        iterations=iterations+1

    # TensorBoard summary logging
    # writer.add_scalar('training/train_precision@1', top1.avg, epoch)
    # writer.add_scalar('training/train_class_loss', cl_losses.avg, epoch)
    # writer.add_scalar('training/train_average_loss', losses.avg, epoch)
    # writer.add_scalar('training/train_KLD', kld_losses.avg, epoch)

    # print(' * Train: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))

    return best_acc,best_model,best_optimum


def train_joint(Dataset, model, criterion, epoch, optimizer, writer, device, args):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int), denoising_noise_value (float) and var_beta (float).
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    labeled_data = read_data(Dataset)
    if args.cuda:
        print("came to cud")
        model = model.cuda()
    # Create instances to accumulate losses etc.
    class_losses = AverageMeter()
    recon_losses = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    lr_change = [int(len(Dataset)*200),int(len(Dataset)*250),int(len(Dataset)*300)]#self.args.train_iterations // 4
    total=args.epochs*args.batch_size
    for iter_count in range(0,total):
        if (iter_count in lr_change):
            print("Came for learning rate change",iter_count)
            for param in optimizer.param_groups:
                param['lr'] = param['lr'] / 10

        inp, target = next(labeled_data)
        if args.cuda:
            inp = inp.cuda()
            class_target = target.cuda()
            recon_target = inp

        # measure data loading time
        data_time.update(time.time() - end)

        # compute model forward
        class_output, recon_output = model(inp)
        if class_output.size(0)>1:
            dummy=torch.Tensor(class_output.size(0),args.batch_size,args.num_classes)
            print("classamples size", class_output.size(0), class_output.shape)
            print("recons size", recon_output.size(0), recon_output.shape)
            class_output=class_output.view_as(dummy)
            recon_output=recon_output.view_as(dummy)
        # calculate loss
        class_loss, recon_loss = criterion(class_output, class_target, recon_output, recon_target)

        # add the individual loss components together
        loss = class_loss + recon_loss

        # record precision/accuracy and losses
        prec1 = accuracy(class_output, class_target)[0]
        top1.update(prec1.item(), inp.size(0))

        losses.update((class_loss + recon_loss).item(), inp.size(0))
        class_losses.update(class_loss.item(), inp.size(0))
        recon_losses.update(recon_loss.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if iter_count % args.print_freq == 0:
            print('Training: [{0}][{1}/{2}]\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})'.format(
                   epoch+1, iter_count, len(Dataset), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   cl_loss=class_losses, top1=top1, recon_loss=recon_losses))



    # print(' * Train: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))


def train_var_joint(Dataset,validate,test_dataloader, model, criterion, epoch, visdom, args,split,iterations):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int), denoising_noise_value (float) and var_beta (float).
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    labeled_data = read_data(Dataset)
    if args.cuda:
        print("came to cud")
        model = model.cuda()
    # Create instances to accumulate losses etc.
    class_losses = AverageMeter()
    recon_losses = AverageMeter()
    kld_losses = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()

    # switch to train mode

    end = time.time()
    acc=0
    best_acc=0
    # train
    lr_change = [int(len(Dataset)*200),int(len(Dataset)*250),int(len(Dataset)*300)]#self.args.train_iterations // 4
    total=args.epochs*args.batch_size
    for iter_count in range(0,total):
        if (iter_count in lr_change):
            print("Came for learning rate change",iter_count)
            for param in optimizer.param_groups:
                param['lr'] = param['lr'] / 10

        inp, target = next(labeled_data)
        if args.cuda:
            inp = inp.cuda()
            class_target = target.cuda()
            recon_target = inp

        # measure data loading time
        data_time.update(time.time() - end)

        # compute model forward
        #print("shape od the input",inp.shape)
        class_samples, recon_samples, mu, std = model(inp)

        '''
        The below if condition is because when we use the nn.Dataparallel to
        fit the model into both the GPU. The output predictions returns by the model
        are spitted into two tensors instead of one. Like
        [2,64,256] instead od [1,128,256]. 

        For smaller model it's fine. But for bigger model we definetly need to
        use the nn.Dataparallel
        '''
        if class_samples.size(0)>1:
            dummy=torch.Tensor(class_samples.size(0),args.batch_size,args.num_classes)
            print("classamples size", class_samples.size(0), class_samples.shape)
            print("recons size", recon_samples.size(0), recon_samples.shape)
            class_samples=class_samples.view_as(dummy)
            recon_samples=recon_samples.view_as(dummy)
        # calculate loss
        class_loss, recon_loss, kld_loss = criterion(class_samples, class_target, recon_samples, recon_target,
                                                     mu, std, args.device)

        # add the individual loss components together and weight the KL term.
        loss = class_loss + recon_loss + args.var_beta * kld_loss

        # take mean to compute accuracy. Note if variational samples are 1 this only gets rid of a dummy dimension.
        output = torch.mean(class_samples, dim=0)

        # record precision/accuracy and losses
        prec1 = accuracy(output, class_target)[0]
        top1.update(prec1.item(), inp.size(0))

        losses.update((class_loss + recon_loss + kld_loss).item(), inp.size(0))
        class_losses.update(class_loss.item(), inp.size(0))
        recon_losses.update(recon_loss.item(), inp.size(0))
        kld_losses.update(kld_loss.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if iter_count % args.print_freq == 0:
            print('Training: [{0}][{1}/{2}]\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                  'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                   epoch+1, iter_count, len(Dataset), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   cl_loss=class_losses, top1=top1, recon_loss=recon_losses, KLD_loss=kld_losses))

            print(' ** Train: Iterations {} Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(iterations, loss=losses, top1=top1))

            plot_visdom(visdom,iter_count,loss.item(),str(int(split*100))+'_loss','loss')
            #acc, val_loss = validate(test_dataloader, model, criterion, epoch, visdom, args.device, args)
            acc, val_loss=validate(test_dataloader, model, criterion, epoch, visdom, args.device, args)
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(model)
                best_optimum=optimizer.state_dict()
            #acc, loss = validate(test_dataloader, model, criterion, epoch, visdom, args.device, args)
            plot_visdom(visdom,iter_count,acc,str(int(split*100))+'_acc','acc')
    # TensorBoard summary logging
    # writer.add_scalar('training/train_precision@1', top1.avg, epoch)
    # writer.add_scalar('training/train_class_loss', cl_losses.avg, epoch)
    # writer.add_scalar('training/train_average_loss', losses.avg, epoch)
    # writer.add_scalar('training/train_KLD', kld_losses.avg, epoch)

    # print(' * Train: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))
    return best_acc,best_model,best_optimum
    #return iterations,acc,loss
