import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score

import time
import sampler
import copy
import sys
import torch.utils.data as data_utils
import torch.nn.functional as F
import pickle
import avg_meter
import libmr
import visualization
import collections

class Solver:
    def __init__(self, args, test_dataloader,val_dataloader_set1,val_dataloader_set2):
        self.args = args
        self.test_dataloader = test_dataloader
        self.val_dataloader_set1=val_dataloader_set1
        self.val_dataloader_set2=val_dataloader_set2

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.sampler = sampler.AdversarySampler(self.args.budget)
        self.topKselector = sampler.TopKSampler(self.args.budget)


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    def var_loss_function(self,output_samples, target, mu, std, device):
        """
        Computes the loss function consisting of a KL term between approximate posterior and prior and the loss for the
        generative classifier. The number of variational samples is one per default, as specified in the command line parser
        and typically is how VAE models and also our unified model is trained.
        We have added the option to flexibly work with an arbitrary amount of samples.
        Parameters:
            output_samples (torch.Tensor): Mini-batch of var_sample many classification prediction values.
            target (torch.Tensor): Classification targets for each element in the mini-batch.
            mu (torch.Tensor): Encoder (recognition model's) mini-batch of mean vectors.
            std (torch.Tensor): Encoder (recognition model's) mini-batch of standard deviation vectors.
            device (str): Device for computation.
        Returns:
            float: normalized classification loss
            float: normalized KL divergence
        """
        #print("oyttttttt",output_samples.shape,target.shape)
        class_loss = nn.CrossEntropyLoss(reduction='sum')

        # Place-holders for the final loss values over all latent space samples
        cl_losses = torch.zeros(output_samples.size(0)).to(device)

        # numerical value for stability of log computation
        eps = 1e-8

        # loop through each sample for each input and calculate the correspond loss. Normalize the losses.
        for i in range(output_samples.size(0)):
            cl_losses[i] = class_loss(output_samples[i], target) / torch.numel(target)

        # average the loss over all samples per input
        cl = torch.mean(cl_losses, dim=0)

        # Compute the KL divergence, normalized by latent dimensionality
        kld = -0.5 * torch.sum(1 + torch.log(eps + std ** 2) - (mu ** 2) - (std ** 2)) / torch.numel(mu)

        return cl, kld

    def validate_var(self,Dataset, model, criterion, epoch, writer, device, args):
        """
        Evaluates/validates the model
        Parameters:
            Dataset (torch.utils.data.Dataset): The dataset
            model (torch.nn.module): Model to be evaluated/validated
            criterion (torch.nn.criterion): Loss function
            epoch (int): Epoch counter
            writer (tensorboard.SummaryWriter): TensorBoard writer instance
            device (str): device name where data is transferred to
            args (dict): Dictionary of (command line) arguments.
                Needs to contain print_freq (int), epochs (int) and patch_size (int).
        Returns:
            float: top1 precision/accuracy
            float: average loss
        """

        # initialize average meters to accumulate values
        cl_losses =  avg_meter.AverageMeter()
        kld_losses = avg_meter.AverageMeter()
        losses =  avg_meter.AverageMeter()
        batch_time = avg_meter.AverageMeter()
        top1 =  avg_meter.AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()

        # evaluate the entire validation dataset
        with torch.no_grad():
            i=0
            for inp, target in self.test_dataloader:
                i=i+1
                inp = inp.to(device)
                target = target.to(device)

                # compute output
                output_samples, mu, std = model(inp)

                # compute loss
                cl_loss, kld_loss = criterion(output_samples, target, mu, std, device)

                # take mean to compute accuracy
                # (does nothing if there isn't more than 1 sample per input other than removing dummy dimension)
                output = torch.mean(output_samples, dim=0)

                # measure and update accuracy
                prec1 = self.accuracy(output, target)[0]
                top1.update(prec1.item(), inp.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update the respective loss values. To be consistent with values reported in the literature we scale
                # our normalized losses back to un-normalized values.
                # For the KLD this also means the reported loss is not scaled by beta, to allow for a fair comparison
                # across potential weighting terms.
                cl_losses.update(cl_loss.item() * model.num_classes, inp.size(0))
                kld_losses.update(kld_loss.item() * model.latent_dim, inp.size(0))
                losses.update((cl_loss + kld_loss).item(), inp.size(0))

                # Print progress
                if i % args.print_freq == 0:
                    print('Validate: [{0}][{1}/{2}]\t' 
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                        epoch+1, i, len(Dataset.val_loader), batch_time=batch_time, loss=losses, cl_loss=cl_losses,
                        top1=top1, KLD_loss=kld_losses))

        # TensorBoard summary logging
        # writer.add_scalar('validation/val_precision@1', top1.avg, epoch)
        # writer.add_scalar('validation/val_class_loss', cl_losses.avg, epoch)
        # writer.add_scalar('validation/val_average_loss', losses.avg, epoch)
        # writer.add_scalar('validation/val_KLD', kld_losses.avg, epoch)

        print(' * Validation: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))

        return top1.avg, losses.avg

    def train(self, querry_dataloader, val_dataloader, task_model, vae, discriminator, unlabeled_dataloader):
        self.args.train_iterations = (self.args.num_images * self.args.train_epochs) // self.args.batch_size
        lr_change = self.args.train_iterations // 4
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_task_model = optim.SGD(task_model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)


        vae.train()
        discriminator.train()
        task_model.train()

        if self.args.cuda:
            vae = vae.cuda()
            discriminator = discriminator.cuda()
            task_model = task_model.cuda()
        
        best_acc = 0
        for iter_count in range(self.args.train_iterations):
            if iter_count is not 0 and iter_count % lr_change == 0:
                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] / 10
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            preds,_ = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            if iter_count%100==0:
                final_accuracy = self.test(task_model)
                print(iter_count,"--------",final_accuracy)
                


            # VAE step
        #     for count in range(self.args.num_vae_steps):
        #         recon, z, mu, logvar = vae(labeled_imgs)
        #         unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta)
        #         unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
        #         transductive_loss = self.vae_loss(unlabeled_imgs, 
        #                 unlab_recon, unlab_mu, unlab_logvar, self.args.beta)
            
        #         labeled_preds = discriminator(mu)
        #         unlabeled_preds = discriminator(unlab_mu)
                
        #         lab_real_preds = torch.ones(labeled_imgs.size(0))
        #         unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                    
        #         if self.args.cuda:
        #             lab_real_preds = lab_real_preds.cuda()
        #             unlab_real_preds = unlab_real_preds.cuda()

        #         dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
        #                 self.bce_loss(unlabeled_preds, unlab_real_preds)
        #         total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
        #         optim_vae.zero_grad()
        #         total_vae_loss.backward()
        #         optim_vae.step()

        #         # sample new batch if needed to train the adversarial network
        #         if count < (self.args.num_vae_steps - 1):
        #             labeled_imgs, _ = next(labeled_data)
        #             unlabeled_imgs = next(unlabeled_data)

        #             if self.args.cuda:
        #                 labeled_imgs = labeled_imgs.cuda()
        #                 unlabeled_imgs = unlabeled_imgs.cuda()
        #                 labels = labels.cuda()

        #     # Discriminator step
        #     for count in range(self.args.num_adv_steps):
        #         with torch.no_grad():
        #             _, _, mu, _ = vae(labeled_imgs)
        #             _, _, unlab_mu, _ = vae(unlabeled_imgs)
                
        #         labeled_preds = discriminator(mu)
        #         unlabeled_preds = discriminator(unlab_mu)
                
        #         lab_real_preds = torch.ones(labeled_imgs.size(0))
        #         unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

        #         if self.args.cuda:
        #             lab_real_preds = lab_real_preds.cuda()
        #             unlab_fake_preds = unlab_fake_preds.cuda()
                
        #         dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
        #                 self.bce_loss(unlabeled_preds, unlab_fake_preds)

        #         optim_discriminator.zero_grad()
        #         dsc_loss.backward()
        #         optim_discriminator.step()

        #         # sample new batch if needed to train the adversarial network
        #         if count < (self.args.num_adv_steps - 1):
        #             labeled_imgs, _ = next(labeled_data)
        #             unlabeled_imgs = next(unlabeled_data)

        #             if self.args.cuda:
        #                 labeled_imgs = labeled_imgs.cuda()
        #                 unlabeled_imgs = unlabeled_imgs.cuda()
        #                 labels = labels.cuda()

                

        #     if iter_count % 100 == 0:
        #         print('Current training iteration: {}'.format(iter_count))
        #         print('Current task model loss: {:.4f}'.format(task_loss.item()))
        #         print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
        #         print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))

        #     if iter_count % 1000 == 0:
        #         acc = self.validate(task_model, val_dataloader)
        #         if acc > best_acc:
        #             best_acc = acc
        #             best_model = copy.deepcopy(task_model)
                
        #         print('current step: {} acc: {}'.format(iter_count, acc))
        #         print('best acc: ', best_acc)


        # if self.args.cuda:
        #     best_model = best_model.cuda()

        #final_accuracy = self.test(best_model)
        return final_accuracy, vae, discriminator


    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader):
        querry_indices = self.sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, 
                                             self.args.cuda)

        return querry_indices
                

    def validate(self, task_model, loader):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels, _ in loader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds,fcOut = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100
    
    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds,fcOut = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100


    def varTest(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds,mu,std = task_model(imgs)
                preds=preds[0]
            
            
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100


    def collectFeatures(self, task_model, loader):
        task_model.eval()
        features=torch.Tensor()
        flables=torch.IntTensor()
        for imgs, labels, _ in loader:
            flables=torch.cat([flables,labels.type(torch.IntTensor)])#.shape
            if self.args.cuda:
                imgs = imgs.cuda()
            with torch.no_grad():
                preds,fcOut = task_model(imgs)
                features=torch.cat([features,fcOut.to('cpu')])#.shape
        return features,flables

    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD


    def accuracy(self,output, target, topk=(1,)):
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

    def plot_visdom(self,vis,x,y,winName,plotName):
        options = dict(fillarea=False,width=400,height=400,xlabel='Iteration',ylabel='Loss',title=winName)
        if (vis.win_exists(winName)==False):
            win = vis.line(X=np.array([0]),Y=np.array([0]),win=winName,name=plotName,opts=options)
            #vis.line(X=np.array([x]),Y=np.array([y]),win=win,update='append')
        else:
            vis.line(X=np.array([x]),Y=np.array([y]),win=winName,update='append',name=plotName)
    def AL_Train(self, querry_dataloader, 
                        val_dataloader, 
                        task_model,
                        vae,
                        MetaModel, 
                        unlabeled_dataloader,
                        split,
                        Flags,
                        visdom):

        self.args.train_iterations = (self.args.num_images * self.args.train_epochs) // self.args.batch_size
        lr_change = [int(len(querry_dataloader)*150),int(len(querry_dataloader)*250),int(len(querry_dataloader)*350)]#self.args.train_iterations // 4
        if self.args.initial_budget==1000:
            lr_change=[3000,4000,8000]

        print("Learning Rate Changes are",lr_change)
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)
        iterations= (self.args.log_interval) * round(lr_change[2]/self.args.log_interval)+1#int(len(querry_dataloader)*128*1.5)#10000#self.args.train_iterations
        print("Now of iterations is",iterations)
        num_adv_steps=15000#self.args.num_adv_steps
        unlabled_dataset_evaluate=[]
        acc=0
        best_acc=0
        # if split>0.1:
        #     iterations=10000
        #     num_adv_steps=12000


        optim_task_model = optim.SGD(task_model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
        optim_discriminator = optim.Adam(MetaModel.parameters(), lr=0.0001)
        optim_vae = torch.optim.Adam(vae.parameters(),lr=0.001, weight_decay=self.args.weight_decay)
        curr_dir=Flags[str(int(split*100))]
        # if os.path.exists(curr_dir):
        #     if len(os.listdir(curr_dir))>0:
        #         nets= [f for f in os.listdir(curr_dir) if f.endswith('.pt')]
        #         nets.sort(reverse=True)
        #         if len(nets)>0:
        #             iterations=0
        #             task_model.load_state_dict(torch.load(curr_dir+str(nets[0])))
        #             print("successfuly Loaded",curr_dir,nets[0])
        #             print("Skipped the current slipt",split)
        # if self.args.reload:
        #     task_model.load_state_dict(torch.load(curr_dir+str(nets[0])))
        MetaModel.train()
        task_model.train()
        vae.train()
        # if split==0.1:
        #     iterations=0
        #     print("Reolading the et")
        #     vae.load_state_dict(torch.load('save_path/best.pt'))
        if self.args.cuda:
            MetaModel = MetaModel.cuda()
            task_model = task_model.cuda()
            vae=vae.cuda()
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            if torch.cuda.device_count()<1:
                # task_model= nn.DataParallel(task_model)
                # MetaModel=nn.DataParallel(MetaModel)
                vae=nn.DataParallel(vae)
        

            # Create instances to accumulate losses etc.
        cl_losses = avg_meter.AverageMeter()
        kld_losses = avg_meter.AverageMeter()
        losses = avg_meter.AverageMeter()
        batch_time = avg_meter.AverageMeter()
        data_time = avg_meter.AverageMeter()
        top1 = avg_meter.AverageMeter()
        end = time.time()
        for iter_count in range(iterations):
            if (iter_count in lr_change):
                print("Came for learning rate change",iter_count)
                for param in optim_vae.param_groups:
                    param['lr'] = param['lr'] / 10
            labeled_imgs, labels = next(labeled_data)
            #print("shape of labeled images",labeled_imgs.shape)
            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                labels = labels.cuda()
            output_samples, mu, std = vae(labeled_imgs)
            #print("shape of labeled images",output_samples.shape,mu.shape,std.shape)
             # calculate loss
            cl_loss, kld_loss = self.var_loss_function(output_samples, labels, mu, std, self.args.device)
            # add the individual loss components together and weight the KL term.
            loss = cl_loss# + self.args.var_beta * kld_loss
            # take mean to compute accuracy. Note if variational samples are 1 this only gets rid of a dummy dimension.
            output = torch.mean(output_samples, dim=0)
            # record precision/accuracy and losses
            prec1 = self.accuracy(output, labels)[0]
            top1.update(prec1.item(), labeled_imgs.size(0))
            losses.update((cl_loss + kld_loss).item(), labeled_imgs.size(0))
            cl_losses.update(cl_loss.item(), labeled_imgs.size(0))
            kld_losses.update(kld_loss.item(), labeled_imgs.size(0))
            # compute gradient and do SGD step
            optim_vae.zero_grad()
            loss.backward()
            optim_vae.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print progress
            if iter_count % self.args.log_interval == 0:
                print('Training: [{0}][{1}/{2}]\t' 
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                    iter_count+1, iter_count, len(querry_dataloader), batch_time=batch_time,
                    data_time=data_time, loss=losses, cl_loss=cl_losses, top1=top1, KLD_loss=kld_losses))
                self.plot_visdom(visdom,iter_count,loss.item(),str(int(split*100))+'_loss','loss')
                acc, loss = self.validate_var(self.test_dataloader, vae, self.var_loss_function, iter_count, 'writer', self.args.device, self.args)
                
                self.plot_visdom(visdom,iter_count,acc,str(int(split*100))+'_acc','acc')
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(vae)
                    torch.save(vae.state_dict(),os.path.join(Flags[str(int(split*100))], "task_"+str(int(best_acc))+".pt"))
                    #print("best accuracy is",best_acc)
                # acc= self.varTest(vae)
                # print("accuracy is",acc)
        if iterations!=0:
            vae=best_model
            #torch.save(vae.state_dict(),'save_path/best.pt')
            torch.save(vae.state_dict(),os.path.join(Flags[str(int(split*100))], "best_"+str(int(best_acc))+".pt"))
        
        # sys.exit()
        dataset_eval_dict_train = self.eval_var_dataset(vae,val_dataloader, self.args.num_classes, self.args.device,
                                    latent_var_samples=self.args.var_samples, model_var_samples=self.args.model_samples)
        print("Validation accuracy: ",dataset_eval_dict_train["accuracy"])

        unlabled_dataset_evaluate=self.Weibull_Sampler(vae, querry_dataloader, val_dataloader,unlabeled_dataloader)#self.args.num_classes, self.args.device,latent_var_samples=self.args.var_samples, model_var_samples=self.args.model_samples)

        ##########################################################################################################################################
        #                           END OF THE OPENSET METHD
        '''
        Weibullsampler has done in below steps1
        Training:
        1. Pass Entire dataset through the model and collect the correct predictions latent variable(mu,std),entropy(output of out encoders)
        2. Then groupby of the class of the latent variable z. So u get list of size 10 (no of classes)
        3. Take the mean of latent vatiable of each class. Mean vector of length 10 - one for each class
        4. Now calculate the cosine distance between each latent vector to it's class mean
        5. Now take these distances for each class to weibul distrubution and fit it per class.(assuming some tailsize)
        
        Threshold and bound estimation:
        Now do the above steps 1 to 4 for half of the validation set
        Now estimate the threshold using the rest of the validation set

        Testing:
        Now do the steps 1 to 4 on testset
        use the weibull models in the training set and pass each of our testing samples
        based on threshold it will give you
        '''
        ##########################################################################################################################################

        return acc,MetaModel,task_model,unlabled_dataset_evaluate
        ##########################################################################################################################################
        #                           END OF THE OPENSET METHD
        ##########################################################################################################################################
        best_acc = 0
        print("Total No of Iterations are",self.args.train_iterations)
        t0 = time.time()
        for iter_count in range(0):#iterations):
            
            if iter_count is not 0 and iter_count % lr_change == 0:
                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] / 10
            labeled_imgs, labels = next(labeled_data)
            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            preds,fcOuts = task_model(labeled_imgs)
            # print("fcOuts",fcOuts.shape)
            # sys.exit()
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            if iter_count % (self.args.log_interval) == 0:
                print('Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(iter_count, iter_count, self.args.batch_size*len(querry_dataloader),task_loss.item()))
                # print("Finished an Iteration")
                acc = self.test(task_model)
                print('Current learning rate: {}. Time taken for epoch: {:0.2f} seconds.Accuracy is {}\n'.format(optim_task_model.param_groups[0]['lr'], time.time() - t0,acc))
                if self.args.visdom:
                    self.plot_visdom(visdom,iter_count,task_loss.item(),str(int(split*100))+'_loss','loss')
                    self.plot_visdom(visdom,iter_count,acc,str(int(split*100))+'_acc','acc')
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(task_model)
                    torch.save(task_model.state_dict(),os.path.join(Flags[str(int(split*100))], "task_"+str(int(acc))+".pt"))
                t0=time.time()

                    # if iter_count % 1000 == 0:
        #     acc = self.validate(task_model, val_dataloader)
        #     if acc > best_acc:
        #         best_acc = acc
        #         best_model = copy.deepcopy(task_model)
            
        #     print('current step: {} acc: {}'.format(iter_count, acc))
        #     print('best acc: ', best_acc)
        
        #torch.save(task_model.module.state_dict(),os.path.join(self.args.out_path, "task_model"+str(iter_count)+".pt"))
                # Discriminator step
        #

        acc = self.test(task_model)
        print("Final Accuracy is ",acc)
        task_model.eval()
        realLoss=[]
        fakeLoss=[]
        t1 = time.time()
        for count in range(0):#num_adv_steps):
            optim_discriminator.zero_grad()
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)  
            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda() 
            with torch.no_grad():
                _lab,labeledFeatures =task_model(labeled_imgs)
                _unlab,unlabFeatures= task_model(unlabeled_imgs)
            labeled_preds = MetaModel(labeledFeatures)
            unlabeled_preds = MetaModel(unlabFeatures)
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            if self.args.cuda:
                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()
            
            real_loss=self.bce_loss(labeled_preds, lab_real_preds)
            fake_loss= self.bce_loss(unlabeled_preds, unlab_fake_preds)
            dsc_loss = real_loss + fake_loss
            dsc_loss.backward()
            optim_discriminator.step() 
            #print( torch.argmax(labeled_preds, dim=1).cpu().numpy())
            #print("unlabeled_preds ara",labeled_preds)
            #x1=accuracy_score(lab_real_preds.cpu().data.numpy(), torch.argmax(labeled_preds, dim=1).cpu().numpy(), normalize=False)
            #x2=accuracy_score(unlab_fake_preds.cpu().data.numpy(), torch.argmax(unlabeled_preds, dim=1).cpu().numpy(), normalize=False)
            #print('-----',MetaModel.net[0].weight[0][0:20])
            if count%500==0:
                if self.args.visdom:
                    self.plot_visdom(visdom,count,real_loss.item(),str(int(split*100))+'_disc','real')
                    self.plot_visdom(visdom,count,fake_loss.item(),str(int(split*100))+'_disc','fake')
                else:
                    print("real loss is",real_loss.item(),fake_loss.item())
                #print("count and loss is",count,dsc_loss.item(),real_loss.item(),fake_loss.item())#,x1,x2)


                
                # sample new batch if needed to train the adversarial network
                # if count < (self.args.num_adv_steps - 1):
                #     labeled_imgs, _ = next(labeled_data)
                #     unlabeled_imgs = next(unlabeled_data)

                #     if self.args.cuda:
                #         labeled_imgs = labeled_imgs.cuda()
                #         unlabeled_imgs = unlabeled_imgs.cuda()
                #         labels = labels.cuda()

        #sys.exit()
        # labeledFeautes,labeledTargets=self.collectFeatures(task_model,querry_dataloader)
        # unLabeledFeautes,unLabeledTargets=self.collectFeatures(task_model,unlabeled_dataloader)
        # MeaFeatures=torch.cat([labeledFeautes,unLabeledFeautes])
        # lab_real_preds = torch.ones(labeledTargets.size(0))
        # unlab_real_preds = torch.zeros(unLabeledTargets.size(0))
        # MetaLabels=torch.cat([lab_real_preds,unlab_real_preds])
        # Metatrain = data_utils.TensorDataset(MeaFeatures, torch.Tensor(MetaLabels))
        # Metatrain_loader = data_utils.DataLoader(Metatrain, batch_size=128, shuffle=True)

        # # MetaModel step
        # print("Metamodel steps are",self.args.num_ML_steps)
        # for count in range(self.args.num_ML_steps):
        #     unsup_loss=0.0
        #     for Meta_batch_idx,(data,target)  in enumerate(Metatrain_loader):
        #         optim_discriminator.zero_grad()
        #         if self.args.cuda:
        #             data, target = data.cuda(), target.cuda()
        #         output=MetaModel(data)
        #         loss=self.bce_loss(output, target)
                
        #         unsup_loss = unsup_loss+loss
        #         if Meta_batch_idx%10==0:
        #             print("unsup loss is",loss.item())
        #         loss.backward() 
        #         optim_discriminator.step()
                

        # # sample new batch if needed to train the adversarial network
        # if count < (self.args.num_vae_steps - 1):
        #     labeled_imgs, _ = next(labeled_data)
        #     unlabeled_imgs = next(unlabeled_data)

        #     if self.args.cuda:
        #         labeled_imgs = labeled_imgs.cuda()
        #         unlabeled_imgs = unlabeled_imgs.cuda()
        #         labels = labels.cuda()

        # if iter_count % 100 == 0:
        #     print('Current training iteration: {}'.format(iter_count))
        #     print('Current task model loss: {:.4f}'.format(task_loss.item()))
        #     print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
        #     print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))

        # if iter_count % 1000 == 0:
        #     acc = self.validate(task_model, val_dataloader)
        #     if acc > best_acc:
        #         best_acc = acc
        #         best_model = copy.deepcopy(task_model)
            
        #     print('current step: {} acc: {}'.format(iter_count, acc))
        #     print('best acc: ', best_acc)


        # if self.args.cuda:
        #     best_model = best_model.cuda()

        # final_accuracy = self.test(best_model)
        print("total time took for MetaModel to finish is",time.time()-t1)
        return acc,MetaModel,task_model,unlabled_dataset_evaluate


    def Weibull_Sampler(self,model,train_loader,val_loader,unlabeled_dataloader):

        dataset_eval_dict_train = self.eval_var_dataset(model, train_loader, self.args.num_classes, self.args.device,latent_var_samples=self.args.var_samples, model_var_samples=self.args.model_samples)
        print("Training accuracy: ", dataset_eval_dict_train["accuracy"])#"accuracy"])
        #Start Preparing for the sampling
        # Get the mean of z for correctly classified data inputs
        mean_zs = self.get_means(dataset_eval_dict_train["zs_correct"])

        # visualize the mean z vectors
        mean_zs_tensor = torch.stack(mean_zs, dim=0)
        # visualize_means(mean_zs_tensor, num_classes, args.dataset, save_path, "z")

        # calculate each correctly classified example's distance to the mean z
        distances_to_z_means_correct_train = self.calc_distances_to_means(mean_zs, dataset_eval_dict_train["zs_correct"],self.args.distance_function)

        #Weibull fitting
        # set tailsize according to command line parameters (according to percentage of dataset size)
        tailsize = int(len(train_loader)*128 * self.args.openset_weibull_tailsize / self.args.num_classes)
        print("Fitting Weibull models with tailsize: " + str(tailsize),len(train_loader))
        tailsizes = [tailsize] * self.args.num_classes
        weibull_models, valid_weibull = self.fit_weibull_models(distances_to_z_means_correct_train, tailsizes)
        # ------------------------------------------------------------------------------------------
        # Fitting on train dataset complete. Determine rejection thresholds/priors on the created split set
        # ------------------------------------------------------------------------------------------
        print("Evaluating original threshold split dataset: " + self.args.dataset + ". This may take a while...")
        threshset_eval_dict = self.eval_var_dataset(model, self.val_dataloader_set1, self.args.num_classes, self.args.device,
                                        latent_var_samples=self.args.var_samples, model_var_samples=self.args.model_samples)

        # Again calculate distances to mean z
        print("Split set accuracy: ", threshset_eval_dict["accuracy"])
        distances_to_z_means_threshset = self.calc_distances_to_means(mean_zs, threshset_eval_dict["zs_correct"],
                                                                self.args.distance_function)
        # get Weibull outlier probabilities for thresh set
        outlier_probs_threshset = self.calc_outlier_probs(weibull_models, distances_to_z_means_threshset)
        threshset_classification = self.calc_openset_classification(outlier_probs_threshset, self.args.num_classes,
                                                            num_outlier_threshs=100)
        #print("threshset_classification is",threshset_classification)
        # also check outlier detection based on entropy
        max_entropy = np.max(threshset_eval_dict["out_entropy"])
        print("Max entopy is",max_entropy)
        threshset_entropy_classification = self.calc_entropy_classification(threshset_eval_dict["out_entropy"],
                                                                    max_entropy,
                                                                    num_outlier_threshs=100)
        #print("calc_entropy_classification",threshset_entropy_classification)
        # determine rejection priors based on 5% of the split data considered as inlying
        if (np.array(threshset_classification["outlier_percentage"]) <= 0.05).any() == True:
            EVT_prior_index = np.argwhere(np.array(threshset_classification["outlier_percentage"])
                                        <= 0.05)[0][0]
            EVT_prior = threshset_classification["thresholds"][EVT_prior_index]
        else:
            EVT_prior = 0.5
            EVT_prior_index = 50

        if (np.array(threshset_entropy_classification["entropy_outlier_percentage"]) <= 0.05).any() == True:
            entropy_threshold_index = np.argwhere(np.array(threshset_entropy_classification["entropy_outlier_percentage"])
                                                <= 0.05)[0][0]
            entropy_threshold = threshset_entropy_classification["entropy_thresholds"][entropy_threshold_index]
        else:
            # this should never actually happen
            entropy_threshold = np.median(threshset_entropy_classification["entropy_thresholds"])
            entropy_threshold_index = 50
        
        
        # ------------------------------------------------------------------------------------------
        # We evaluate the validation set to later evaluate trained dataset's statistical inlier/outlier estimates.
        print("Evaluating original validation dataset: " + self.args.dataset + ". This may take a while...")
        dataset_eval_dict = self.eval_var_dataset(model, self.val_dataloader_set2, self.args.num_classes, self.args.device,latent_var_samples=self.args.var_samples, model_var_samples=self.args.model_samples)

        # Again calculate distances to mean z
        print("Validation accuracy: ", dataset_eval_dict["accuracy"])
        distances_to_z_means_correct = self.calc_distances_to_means(mean_zs, dataset_eval_dict["zs_correct"],self.args.distance_function)

        # Evaluate outlier probability of trained dataset's validation set
        outlier_probs_correct = self.calc_outlier_probs(weibull_models, distances_to_z_means_correct)

        dataset_classification_correct = self.calc_openset_classification(outlier_probs_correct, self.args.num_classes,
                                                                    num_outlier_threshs=100)
        dataset_entropy_classification_correct = self.calc_entropy_classification(dataset_eval_dict["out_entropy"],
                                                                            max_entropy,
                                                                            num_outlier_threshs=100)

        print(self.args.dataset + '(trained) EVT outlier percentage: ' +str(dataset_classification_correct["outlier_percentage"][EVT_prior_index]))
        print(self.args.dataset + '(trained) entropy outlier percentage: ' +str(dataset_entropy_classification_correct["entropy_outlier_percentage"][entropy_threshold_index]))
        
        ##########################################################################################################################################
        #                           START ON THE RESt OF TRAINING SET
        ##########################################################################################################################################
        od=0
        openset_dataset='cifar10'
        openset_datasets_names = self.args.openset_datasets.strip().split(',')
        openset_sampler_methods=self.args.samplerMethod.strip().split(',')
        samplerMethod=self.args.sampler
        print("The Sampling Method is",samplerMethod)
        openset_datasets = []
        # Repeat process for open set recognition on unseen datasets (
        openset_dataset_eval_dicts = collections.OrderedDict()
        openset_outlier_probs_dict = collections.OrderedDict()
        openset_classification_dict = collections.OrderedDict()
        openset_entropy_classification_dict = collections.OrderedDict()
        print("Evaluating on rest of the tain set This may take a while...",len( self.test_dataloader)*128)
        openset_dataset_eval_dict = self.eval_var_openset_dataset(model,unlabeled_dataloader, self.args.num_classes,
                                                         self.args.device, latent_var_samples=self.args.var_samples,
                                                         model_var_samples=self.args.model_samples)
        


        #with open('openset_dataset_eval_dict', 'wb') as fp:pickle.dump(openset_dataset_eval_dict, fp)
        #with open('mean_zs', 'wb') as fp:pickle.dump(mean_zs, fp)
        # with open('weibull_models', 'wb') as fp:pickle.dump(weibull_models, fp)
        #with open('openset_distances_to_z_means', 'wb') as fp:pickle.dump(openset_distances_to_z_means, fp)
        # sys.exit()
        openset_distances_to_z_means = self.calc_distances_to_means(mean_zs, openset_dataset_eval_dict["zs"],'cosine')#self.args.distance_function)
    
        openset_outlier_probs = self.calc_outlier_probs(weibull_models, openset_distances_to_z_means)

        # getting outlier classification accuracies across the entire datasets
        openset_classification = self.calc_openset_classification(openset_outlier_probs, self.args.num_classes,num_outlier_threshs=100)

        openset_entropy_classification = self.calc_entropy_classification(openset_dataset_eval_dict["out_entropy"],max_entropy, num_outlier_threshs=100)

        if samplerMethod=='classifierProbability':
            topk=self.topKselector.sample([],openset_dataset_eval_dict['querry_pool_indices'],'NONE',self.args.budget)
        elif  samplerMethod=='LatentMeanDistance':
            Rvalues=[]
            Rindexes=[]
            for i in range(0,self.args.num_classes):#openset_outlier_probs:
                Rvalues.append(torch.Tensor(openset_distances_to_z_means[i]))
                Rindexes.append(torch.Tensor(openset_dataset_eval_dict['collect_indexes_per_class'][i]))
            if  self.args.samplePerClass:
                print("came to sample per class")
                perclassSample=[]
                for i in range(0,self.args.num_classes):
                    topk=self.topKselector.sample(Rvalues[i],Rindexes[i],samplerMethod,int(self.args.budget/self.args.num_classes))
                    perclassSample.append(torch.Tensor(topk))

                topk=torch.cat(perclassSample,dim=0)
            else:
                topk=self.topKselector.sample(torch.cat(Rvalues, dim=0),torch.cat(Rindexes, dim=0),samplerMethod,self.args.budget)
                
        elif  samplerMethod=='WiebullOutlierProbs':
            Rvalues=[]
            Rindexes=[]
            for i in range(0,self.args.num_classes):#openset_outlier_probs:
                Rvalues.append(torch.Tensor(openset_outlier_probs[i]))
                Rindexes.append(torch.Tensor(openset_dataset_eval_dict['collect_indexes_per_class'][i]))
            if  self.args.samplePerClass:
                print("came to  per class")
                perclassSample=[]
                for i in range(0,self.args.num_classes):
                    topk=self.topKselector.sample(Rvalues[i],Rindexes[i],samplerMethod,int(self.args.budget/self.args.num_classes))
                    perclassSample.append(torch.Tensor(topk))

                topk=torch.cat(perclassSample,dim=0)
            else:
                topk=self.topKselector.sample(torch.cat(Rvalues, dim=0),torch.cat(Rindexes, dim=0),samplerMethod,self.args.budget)
                
        elif samplerMethod=='Entropy':
            topk=self.topKselector(openset_dataset_eval_dict,openset_classification,samplerMethod)
        elif samplerMethod=='openSet':
            topk=self.topKselector(openset_dataset_eval_dict,openset_entropy_classification,samplerMethod)

        # dictionary of dictionaries: per datasetname one dictionary with respective values
        openset_dataset_eval_dicts[openset_datasets_names[od]] = openset_dataset_eval_dict
        openset_outlier_probs_dict[openset_datasets_names[od]] = openset_outlier_probs
        openset_classification_dict[openset_datasets_names[od]] = openset_classification
        openset_entropy_classification_dict[openset_datasets_names[od]] = openset_entropy_classification

        # print outlier rejection values for all unseen unknown datasets
        for other_data_name, other_data_dict in openset_classification_dict.items():
            print(other_data_name + ' EVT outlier percentage: ' +
                str(other_data_dict["outlier_percentage"][entropy_threshold_index]))

        for other_data_name, other_data_dict in openset_entropy_classification_dict.items():
            print(other_data_name + ' entropy outlier percentage: ' +
                str(other_data_dict["entropy_outlier_percentage"][entropy_threshold_index]))
            

        # joint prediction uncertainty plot for all datasets
        if (self.args.train_var and self.args.var_samples > 1) or self.args.model_samples > 1:
            visualization.visualize_classification_uncertainty(dataset_eval_dict["out_mus_correct"],
                                                dataset_eval_dict["out_sigmas_correct"],
                                                openset_dataset_eval_dicts,
                                                "out_mus", "out_sigmas",
                                                self.args.dataset + ' (trained)',
                                                self.args.var_samples, 'save_path')

        # visualize the outlier probabilities
        visualization.visualize_weibull_outlier_probabilities(outlier_probs_correct, openset_outlier_probs_dict,
                                                self.args.dataset + ' (trained)', 'save_path', tailsize)

        visualization.visualize_classification_scores(dataset_eval_dict["out_mus_correct"], openset_dataset_eval_dicts, 'out_mus',
                                        self.args.dataset + ' (trained)', 'save_path')

        visualization.visualize_entropy_histogram(dataset_eval_dict["out_entropy"], openset_dataset_eval_dicts,
                                    dataset_entropy_classification_correct["entropy_thresholds"][-1], "out_entropy",
                                    self.args.dataset + ' (trained)', 'save_path')

        # joint plot for outlier detection accuracy for seen and both unseen datasets
        visualization.visualize_openset_classification(dataset_classification_correct["outlier_percentage"],
                                        openset_classification_dict, "outlier_percentage",
                                        self.args.dataset + ' (trained)',
                                        dataset_classification_correct["thresholds"], 'save_path', tailsize)

        visualization.visualize_entropy_classification(dataset_entropy_classification_correct["entropy_outlier_percentage"],
                                        openset_entropy_classification_dict, "entropy_outlier_percentage",
                                        self.args.dataset + ' (trained)',
                                        dataset_entropy_classification_correct["entropy_thresholds"], 'save_path')

        return topk

    def eval_var_dataset(self,model, data_loader, num_classes, device, latent_var_samples=1, model_var_samples=1):
        """
        Evaluates an entire dataset with the variational or joint model and stores z values, latent mus and sigmas and
        output predictions according to whether the classification is correct or not.
        The values for correct predictions can later be used for plotting or fitting of Weibull models.
        Parameters:
            model (torch.nn.module): Trained model.
            data_loader (torch.utils.data.DataLoader): The dataset loader.
            num_classes (int): Number of classes.
            device (str): Device to compute on.
            latent_var_samples (int): Number of latent space variational samples.
            model_var_samples (int): Number of variational samples of the entire model, e.g. MC dropout.
        Returns:
            dict: Dictionary of results and latent values, separated by whether the classification was correct or not.
        """

        # switch to evaluation mode unless MC dropout is active
        if model_var_samples > 1:
            model.train()
        else:
            model.eval()

        correctly_identified = 0
        tot_samples = 0

        out_mus_correct = []
        out_sigmas_correct = []
        out_mus_false = []
        out_sigmas_false = []
        encoded_mus_correct = []
        encoded_mus_false = []
        encoded_sigmas_correct = []
        encoded_sigmas_false = []
        zs_correct = []
        zs_false = []

        out_entropy = []
        collect_indexes=[]
    
        for i in range(num_classes):
            out_mus_correct.append([])
            out_mus_false.append([])
            out_sigmas_correct.append([])
            out_sigmas_false.append([])
            encoded_mus_correct.append([])
            encoded_mus_false.append([])
            encoded_sigmas_correct.append([])
            encoded_sigmas_false.append([])
            zs_false.append([])
            zs_correct.append([])
            collect_indexes.append([])

            
        # evaluate the encoder and classifier and store results in corresponding lists according to predicted class.
        # Prediction mean confidence and uncertainty is also obtained if amount of latent samples is greater than one.
        with torch.no_grad():
            for j, (inputs, classes,_) in enumerate(data_loader):
                inputs, classes = inputs.to(device), classes.to(device)

                out_samples = torch.zeros(model_var_samples, latent_var_samples, inputs.size(0), num_classes).to(device)
                z_samples = torch.zeros(model_var_samples, latent_var_samples,
                                        inputs.size(0), model.latent_dim).to(device)

                # sampling the model, then z and classifying
                for k in range(model_var_samples):
                    encoded_mu, encoded_std = model.encode(inputs)#.module.

                    for i in range(latent_var_samples):
                        z = model.reparameterize(encoded_mu, encoded_std)
                        z_samples[k][i] = z

                        cl = model.classifier(z)
                        out = torch.nn.functional.softmax(cl, dim=1)
                        out_samples[k][i] = out

                out_mean = torch.mean(torch.mean(out_samples, dim=0), dim=0)
                if model_var_samples > 1:
                    out_std = torch.std(torch.mean(out_samples, dim=0), dim=0)
                else:
                    out_std = torch.squeeze(torch.std(out_samples, dim=1))

                zs_mean = torch.mean(torch.mean(z_samples, dim=0), dim=0)
                
                # calculate entropy for the means of samples: - sum pc*log(pc)
                eps = 1e-10
                out_entropy.append(-torch.sum(out_mean*torch.log(out_mean + eps), dim=1).cpu().detach().numpy())

                # for each input and respective prediction store independently depending on whether classification was
                # correct. The list of correct classifications is later used for fitting of Weibull models if the
                # data_loader is loading the training set.
                for i in range(inputs.size(0)):
                    tot_samples += 1
                    idx = torch.argmax(out_mean[i]).item()
                    if classes[i].item() != idx:
                        out_mus_false[idx].append(out_mean[i][idx].item())
                        out_sigmas_false[idx].append(out_std[i][idx].item())
                        encoded_mus_false[idx].append(encoded_mu[i].data)
                        encoded_sigmas_false[idx].append(encoded_std[i].data)
                        zs_false[idx].append(zs_mean[i].data)
                    else:
                        correctly_identified += 1
                        out_mus_correct[idx].append(out_mean[i][idx].item())
                        out_sigmas_correct[idx].append(out_std[i][idx].item())
                        encoded_mus_correct[idx].append(encoded_mu[i].data)
                        encoded_sigmas_correct[idx].append(encoded_std[i].data)
                        zs_correct[idx].append(zs_mean[i].data)
                    

        acc = correctly_identified / float(tot_samples)


        # d={"accuracy": acc, "encoded_mus_correct": encoded_mus_correct, "encoded_mus_false": encoded_mus_false,
        #         "encoded_sigmas_correct": encoded_sigmas_correct, "encoded_sigmas_false": encoded_sigmas_false,
        #         "zs_correct": zs_correct, "zs_false": zs_false,
        #         "out_mus_correct": out_mus_correct, "out_sigmas_correct": out_sigmas_correct,
        #         "out_mus_false": out_mus_false, "out_sigmas_false": out_sigmas_false,
        #         "out_entropy": out_entropy}

        # with open('outfile', 'wb') as fp:pickle.dump(d, fp)
        # print('--------------',zs_correct)
        # stack list of tensors into tensors
        for i in range(len(encoded_mus_correct)):
            if len(encoded_mus_correct[i]) > 0:
                encoded_mus_correct[i] = torch.stack(encoded_mus_correct[i], dim=0)
                encoded_sigmas_correct[i] = torch.stack(encoded_sigmas_correct[i], dim=0)
                zs_correct[i] = torch.stack(zs_correct[i], dim=0)
                if(len(zs_correct[i])==0):
                    print("came here")
            if len(encoded_mus_false[i]) > 0:
                encoded_mus_false[i] = torch.stack(encoded_mus_false[i], dim=0)
                encoded_sigmas_false[i] = torch.stack(encoded_sigmas_false[i], dim=0)
                zs_false[i] = torch.stack(zs_false[i], dim=0)

        out_entropy = np.concatenate(out_entropy).ravel().tolist()
        
        # d={"accuracy": acc, "encoded_mus_correct": encoded_mus_correct, "encoded_mus_false": encoded_mus_false,
        #         "encoded_sigmas_correct": encoded_sigmas_correct, "encoded_sigmas_false": encoded_sigmas_false,
        #         "zs_correct": zs_correct, "zs_false": zs_false,
        #         "out_mus_correct": out_mus_correct, "out_sigmas_correct": out_sigmas_correct,
        #         "out_mus_false": out_mus_false, "out_sigmas_false": out_sigmas_false,
        #         "out_entropy": out_entropy}

        # with open('outfile2', 'wb') as fp:pickle.dump(d, fp)
        # Return a dictionary containing all the stored values
        return {"accuracy": acc, 
                "encoded_mus_correct": encoded_mus_correct, 
                "encoded_mus_false": encoded_mus_false,
                "encoded_sigmas_correct": encoded_sigmas_correct, 
                "encoded_sigmas_false": encoded_sigmas_false,
                "zs_correct": zs_correct, 
                "zs_false": zs_false,
                "out_mus_correct": out_mus_correct, 
                "out_sigmas_correct": out_sigmas_correct,
                "out_mus_false": out_mus_false, 
                "out_sigmas_false": out_sigmas_false,
                "out_entropy": out_entropy}

    def get_means(self,tensors_list):
        """
        Calculate the mean of a list of tensors for each tensor in the list. In our case the list typically contains
        a tensor for each class, such as the per class z values.
        Parameters:
            tensors_list (list): List of Tensors
        Returns:
            list: List of Tensors containing mean vectors
        """

        means = []
        for i in range(len(tensors_list)):
            if isinstance(tensors_list[i], torch.Tensor):
                means.append(torch.mean(tensors_list[i], dim=0))
            else:
                means.append([])

  
        return means


    def calc_distances_to_means(self,means, tensors, distance_function='cosine'):
        """
        Function to calculate distances between tensors, in our case the mean zs per class and z for each input.
        Wrapper around torch.nn.functonal distances with specification of which distance function to choose.
        Parameters:
            means (list): List of length corresponding to number of classes containing torch tensors (typically mean zs).
            tensors (list): List of length corresponding to number of classes containing tensors (typically zs).
            distance_function (str): Specification of distance function. Choice of cosine|euclidean|mix.
        Returns:
            list: List of length corresponding to number of classes containing tensors with distance values
        """

        def distance_func(a, b, w_eucl, w_cos):
            """
            Weighted distance function consisting of cosine and euclidean components.
            Parameters:
                a (torch.Tensor): First tensor.
                b (torch.Tensor): Second tensor.
                w_eucl (float): Weight for the euclidean distance term.
                w_cos (float): Weight for the cosine similarity term.
            """
            d = w_cos * (1 - torch.nn.functional.cosine_similarity(a.view(1, -1), b)) + \
                w_eucl * torch.nn.functional.pairwise_distance(a.view(1, -1), b, p=2)
            return d

        distances = []

        # weight values for individual distance components
        w_eucl = 0.0
        w_cos = 0.0
        if distance_function == 'euclidean':
            w_eucl = 1.0
        elif distance_function == 'cosine':
            w_cos = 1.0
        elif distance_function == 'mix':
            w_eucl = 0.5
            w_cos = 0.5
        else:
            raise ValueError("distance function not implemented")

        # loop through each class in means and calculate the distances with the respective tensor.
        for i in range(len(means)):
            # check for tensor type, e.g. list could be empty
            if isinstance(tensors[i], torch.Tensor) and isinstance(means[i], torch.Tensor):
                distances.append(distance_func(means[i], tensors[i], w_eucl, w_cos))
            else:
                distances.append([])

        return distances


    def fit_weibull_models(self,distribution_values, tailsizes, num_max_fits=50):
        """
        Function to fit weibull models on distribution values per class. The distribution values in our case are the
        distances of an inputs approximate posterior value to the per class mean latent z, i.e. The Weibull model fits
        regions of high density and gives credible intervals.
        The tailsize specifies how many outliers are expected in the dataset for which the model has been trained.
        We use libmr https://github.com/Vastlab/libMR (installable through e.g. pip) for the Weibull model fitting.
        Parameters:
            distribution_values (list): Values on which the fit is conducted. In our case latent space distances.
            tailsizes (list): List of integers, specifying tailsizes per class. For a balanced dataset typically the same.
            num_max_fits (int): Number of attempts to fit the Weibull models before timing out and returning unsuccessfully.
        Returns:
            list: List of Weibull models with their respective parameters (stored in libmr class instances).
        """

        weibull_models = []

        # loop through the list containing distance values per class
        for i in range(len(distribution_values)):
            # for each class set the initial success to False and number of attempts to 0
            is_valid = False
            count = 0

            # If the list contains distance values conduct a fit. If it is empty, e.g. because there is not a single
            # prediction for the corresponding class, continue with the next class. Note that the latter isn't expected for
            # a model that has been trained for even just a short while.
            if isinstance(distribution_values[i], torch.Tensor):
                distribution_values[i] = distribution_values[i].cpu().numpy()
                # weibull model per class
                weibull_models.append(libmr.MR())
                # attempt num_max_fits many fits before aborting
                while is_valid is False and count < num_max_fits:
                    # conduct the fit with libmr
                    weibull_models[i].fit_high(distribution_values[i], tailsizes[i])
                    is_valid = weibull_models[i].is_valid
                    count += 1
                if not is_valid:
                    print("Weibull fit for class " + str(i) + " not successful after " + str(num_max_fits) + " attempts")
                    return weibull_models, False
            else:
                weibull_models.append([])

        return weibull_models, True


    def calc_outlier_probs(self,weibull_models, distances):
        """
        Calculates statistical outlier probability using the weibull models' CDF.
        Note that we have coded this function to loop over each class because we have previously categorized the distances
        into their respective classes already.
        Parameters:
            weibull_models (list): List of libmr class instances containing the Weibull model parameters and functions.
            distances (list): List of per class torch tensors or numpy arrays with latent space distance values.
        Returns:
            list: List of length corresponding to number of classes with outlier probabilities for each respective input.
        """

        outlier_probs = []
        # loop through all classes, i.e. all available weibull models as there is one weibull model per class.
        for i in range(len(weibull_models)):
            # optionally convert the type of the distance vectors
            if isinstance(distances[i], torch.Tensor):
                distances[i] = distances[i].cpu().numpy().astype(np.double)
            elif isinstance(distances[i], list):
                # empty list
                outlier_probs.append([])
                continue
            else:
                distances[i] = distances[i].astype(np.double)

            # use the Weibull models' CDF to evaluate statistical outlier rejection probabilities.
            outlier_probs.append(weibull_models[i].w_score_vector(distances[i]))

        return outlier_probs


    def calc_openset_classification(self,data_outlier_probs, num_classes, num_outlier_threshs=50):
        """
        Calculates the percentage of dataset outliers given a set of outlier probabilities over a range of rejection priors.
        Parameters:
            data_outlier_probs (list): List of outlier probabilities for an entire dataset, categorized by class.
            num_classes (int): Number of classes.
            num_outlier_threshs (int): Number of outlier rejection priors (evenly spread over the interval (0,1)).
        Returns:
            dict: Dictionary containing outlier percentages and corresponding rejection prior values.
        """

        dataset_outliers = []
        threshs = []

        # loop through each rejection prior value and evaluate the percentage of the dataset being considered as
        # statistical outliers, i.e. each data point's outlier probability > rejection prior.
        for i in range(num_outlier_threshs - 1):
            outlier_threshold = (i + 1) * (1.0 / num_outlier_threshs)
            threshs.append(outlier_threshold)

            dataset_outliers.append(0)
            total_dataset = 0

            for j in range(num_classes):
                total_dataset += len(data_outlier_probs[j])

                for k in range(len(data_outlier_probs[j])):
                    if data_outlier_probs[j][k] > outlier_threshold:
                        dataset_outliers[i] += 1

            dataset_outliers[i] = dataset_outliers[i] / float(total_dataset)

        return {"thresholds": threshs, "outlier_percentage": dataset_outliers}


    def calc_entropy_classification(self,dataset_entropies, max_thresh_value, num_outlier_threshs=50):
        """
        Calculates the percentage of dataset outliers given a set of entropies over a range of rejection priors.
        Parameters:
            dataset_entropies (list): List of entropies for the entire dataset (each instance)
            num_outlier_threshs (int): Number of outlier rejection priors (evenly spread over the interval (0,1)).
        Returns:
            dict: Dictionary containing outlier percentages and corresponding rejection prior values.
        """

        dataset_outliers = []
        threshs = []

        total_dataset = float(len(dataset_entropies))

        # loop through each rejection prior value and evaluate the percentage of the dataset being considered as
        # statistical outliers, i.e. each data point's outlier probability > rejection prior.
        for i in range(num_outlier_threshs - 1):
            outlier_threshold = (i + 1) * (max_thresh_value / num_outlier_threshs)
            threshs.append(outlier_threshold)

            dataset_outliers.append(0)

            for k in range(len(dataset_entropies)):
                if dataset_entropies[k] > outlier_threshold:
                    dataset_outliers[i] += 1

            dataset_outliers[i] = dataset_outliers[i] / total_dataset

        return {"entropy_thresholds": threshs, "entropy_outlier_percentage": dataset_outliers}

    def eval_var_openset_dataset(self,model, data_loader, num_classes, device, latent_var_samples=1, model_var_samples=1):
        """
        Evaluates an entire dataset with the variational or joint model and stores z values, latent mus and sigmas and
        output predictions such that they can later be used for statistical outlier evaluation with the fitted Weibull
        models. This is merely for convenience to keep the rest of the code API the same. Note that the Weibull model's
        prediction of whether a sample from an unknown dataset is a statistical outlier or not can be done on an instance
        level. Similar to the eval_dataset function but without splitting of correct vs. false predictions as the dataset
        is unknown in the open-set scenario.
        Parameters:
            model (torch.nn.module): Trained model.
            data_loader (torch.utils.data.DataLoader): The dataset loader.
            num_classes (int): Number of classes.
            device (str): Device to compute on.
            latent_var_samples (int): Number of latent space variational samples.
            model_var_samples (int): Number of variational samples of the entire model, e.g. MC dropout.
        Returns:
            dict: Dictionary of results and latent values.
        """

        # switch to evaluation mode unless MC dropout is active
        if model_var_samples > 1:
            model.train()
        else:
            model.eval()

        out_mus = []
        out_sigmas = []
        encoded_mus = []
        encoded_sigmas = []
        zs = []
        all_preds = []
        all_indices = []

        out_entropy = []
        collect_indexes_per_class=[]

        for i in range(num_classes):
            out_mus.append([])
            out_sigmas.append([])
            encoded_mus.append([])
            encoded_sigmas.append([])
            zs.append([])
            collect_indexes_per_class.append([])

        # evaluate the encoder and classifier and store results in corresponding lists according to predicted class.
        # Prediction mean confidence and uncertainty is also obtained if amount of latent samples is greater than one.
        with torch.no_grad():
            for inputs, classes,indexes in data_loader:
                inputs, classes = inputs.to(device), classes.to(device)

                out_samples = torch.zeros(model_var_samples, latent_var_samples, inputs.size(0), num_classes).to(device)
                z_samples = torch.zeros(model_var_samples, latent_var_samples,
                                        inputs.size(0), model.latent_dim).to(device)

                # sampling the model, then z and classifying
                for k in range(model_var_samples):
                    encoded_mu, encoded_std = model.encode(inputs)

                    for i in range(latent_var_samples):
                        z = model.reparameterize(encoded_mu, encoded_std)
                        z_samples[k][i] = z

                        cl = model.classifier(z)
                        out = torch.nn.functional.softmax(cl, dim=1)
                        out_samples[k][i] = out

                # calculate the mean and std. Only removes a dummy dimension if number of variational samples is set to one.
                out_mean = torch.mean(torch.mean(out_samples, dim=0), dim=0)
                # preds = out_mean.cpu().data
                all_preds.extend(out_mean.cpu().data)
                all_indices.extend(indexes)
                if model_var_samples > 1:
                    out_std = torch.std(torch.mean(out_samples, dim=0), dim=0)
                else:
                    out_std = torch.squeeze(torch.std(out_samples, dim=1))
                zs_mean = torch.mean(torch.mean(z_samples, dim=0), dim=0)
                
                # calculate entropy for the means of samples: - sum pc*log(pc)
                eps = 1e-10
                out_entropy.append(- torch.sum(out_mean*torch.log(out_mean + eps), dim=1).cpu().detach().numpy())

                # In contrast to the eval_dataset function, there is no split into correct or false values as the dataset
                # is unknown.
                for i in range(inputs.size(0)):
                    idx = torch.argmax(out_mean[i]).item()
                    out_mus[idx].append(out_mean[i][idx].item())
                    out_sigmas[idx].append(out_std[i][idx].item())
                    encoded_mus[idx].append(encoded_mu[i].data)
                    encoded_sigmas[idx].append(encoded_std[i].data)
                    zs[idx].append(zs_mean[i].data)
                    collect_indexes_per_class[idx].append(indexes[i].item())

        # stack latent activations into a tensor
        for i in range(len(encoded_mus)):
            if len(encoded_mus[i]) > 0:
                encoded_mus[i] = torch.stack(encoded_mus[i], dim=0)
                encoded_sigmas[i] = torch.stack(encoded_sigmas[i], dim=0)
                zs[i] = torch.stack(zs[i], dim=0)

        out_entropy = np.concatenate(out_entropy).ravel().tolist()

        d={"encoded_mus": encoded_mus, "encoded_sigmas": encoded_sigmas,
                "out_mus": out_mus, "out_sigmas": out_sigmas, "zs": zs,
                "out_entropy": out_entropy,'all_preds':all_preds,    "all_indices":all_indices}

        # with open('outfile', 'wb') as fp:pickle.dump(d, fp)

        all_preds=[]
        for i in range(0,len(d['all_preds'])):
            idx=torch.argmax(d['all_preds'][i]).item()
            all_preds.append(d['all_preds'][i][idx])#.data.item())
        all_preds
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.args.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        # Return a dictionary of stored values.
        # with open('outfile', 'wb') as fp:pickle.dump(d, fp)
        return {"encoded_mus": encoded_mus, "encoded_sigmas": encoded_sigmas,
                "out_mus": out_mus, "out_sigmas": out_sigmas, "zs": zs,
                "out_entropy": out_entropy,'all_preds':all_preds,
                "all_indices":all_indices,
                'querry_pool_indices':querry_pool_indices,
                'collect_indexes_per_class':collect_indexes_per_class}