import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data

import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
import vgg
from solver import Solver
from utils import *
import arguments
import pickle
import visdom
import shutil
import yaml
from easydict import EasyDict
from initialization import WeightInit
vis = visdom.Visdom()
environment='AL_Cifar100'
vis.delete_env(environment) #If you want to clear all the old plots for this python Experiments.Resets the Environment
vis = visdom.Visdom(env=environment)
# Execution flags


def create_flders(splits,args):
    Flags = {}
    Flags['Dir']='task_net_models/'
    Flags['MNT']=args.out_path+'/'
    Flags['sPath']=Flags['MNT']+environment+'/'
    Dir_Use=splits
    savedModels=['imScores','savedModels']


    #remove existing directory
    if os.path.exists(Flags['sPath']) and os.path.isdir(Flags['sPath']):
        shutil.rmtree(Flags['sPath'])
    
    #create the directory
    os.makedirs(Flags['sPath'])
    for directory in Dir_Use:
        if not os.path.exists(Flags['sPath']+str(int(directory*100))):
            os.makedirs(Flags['sPath']+str(int(directory*100)))
            Flags[str(int(directory*100))]=Flags['sPath']+str(int(directory*100))+'/'


    # if not os.path.exists(Flags['sPath']):
    #     os.makedirs(Flags['sPath']+environment)
    return Flags

def config_to_str(config):
    attrs = vars(config)
    string_val = "Config: -----\n"
    string_val += "\n".join("%s: %s" % item for item in attrs.items())
    string_val += "\n----------"
    return string_val

def cifar_transformer():
    return transforms.Compose([
            transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5,],
            #                     std=[0.5, 0.5, 0.5]),
        ])

def main(args):
    if args.dataset == 'cifar10':
        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)
        
        train_dataset = CIFAR10(args.data_path)

        args.num_images = 50000
        args.num_val = 5000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR100(args.data_path)
        print("=========CIFAR 100===============")
        args.num_val = 5000
        args.num_images = 50000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 100

    elif args.dataset == 'imagenet':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = ImageNet(args.data_path)

        args.num_val = 128120
        args.num_images = 1281167
        args.budget = 64060
        args.initial_budget = 128120
        args.num_classes = 1000
    else:
        raise NotImplementedError
    
    if args.sameIndexs:
        print("Setting the random Index")
        random.seed(30)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Config: %s" % config_to_str(args))
    all_indices = set(np.arange(args.num_images))
    val_indices = random.sample(all_indices, args.num_val)
    all_indices = np.setdiff1d(list(all_indices), val_indices)
    initial_indices = random.sample(list(all_indices), args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    #The below lines are only for weibull distrubution which need two validation sets to pll off
    val_indices_set1=val_indices[0:int(len(val_indices)/2)]
    val_indices_set2=val_indices[int(len(val_indices)/2):]
    val_sampler_set1 = data.sampler.SubsetRandomSampler(val_indices)
    val_sampler_set2 = data.sampler.SubsetRandomSampler(val_indices)
    val_dataloader_set1 = data.DataLoader(train_dataset, sampler=val_sampler_set1,
            batch_size=args.batch_size, drop_last=False)
    val_dataloader_set2 = data.DataLoader(train_dataset, sampler=val_sampler_set2,
            batch_size=args.batch_size, drop_last=False)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=args.batch_size, drop_last=False)
    val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler,
            batch_size=args.batch_size, drop_last=False)

    print("length od the Querry loader",len(querry_dataloader)*128)  
    print("length od the Validation loader",len(val_dataloader)*128)  
    args.cuda = torch.cuda.is_available()

    solver = Solver(args, test_dataloader,val_dataloader_set1,val_dataloader_set2)

    splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    current_indices = list(initial_indices)

    accuracies = []

    Flags=create_flders(splits,args)
    print(Flags)
    
    

    for split in splits:
        # need to retrain all the models on the new images
        # re initialize and retrain the models
        task_model = vgg.vgg16_bn(num_classes=args.num_classes)

        # print(task_model)
        # sys.exit()
        #vae = model.VAE(args.latent_dim)
        num_colors=3
        vae=model.WRN(args.device, args.num_classes, num_colors, args)
       
        print()
        # WeightInitializer = WeightInit(args.weight_init)
        # WeightInitializer.init_model(vae)
        MetaModel = model.Discriminator(args.latent_dim)
        #discriminator= model.Discriminator(args.latent_dim)

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)
        
        print("length od the Unlabled loader",len(unlabeled_dataloader)*128)  
       
        # train the models on the current data
        # acc, vae, discriminator = solver.train(querry_dataloader,
        #                                        val_dataloader,
        #                                        task_model, 
        #                                        vae, 
        #                                        MetaModel,
        #                                        unlabeled_dataloader)

        acc, discriminator,trained_taskmodel,topK = solver.AL_Train(querry_dataloader,
                                                                val_dataloader,
                                                                task_model,
                                                                vae,
                                                                MetaModel,
                                                                unlabeled_dataloader,
                                                                split,
                                                                Flags,
                                                                vis)


        print('Final accuracy with {}% of data is: {:.2f}'.format(int(split*100), acc))
        
        accuracies.append(acc)
        print("All accuracies until noe",accuracies)
        if args.method:
            sampled_indices = solver.sample_for_labeling(trained_taskmodel, discriminator, unlabeled_dataloader)
        else:
            sampled_indices=topK#all_statistics['querry_pool_indices']
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=True)

    torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':
    args = arguments.get_args()
    # with open(args.work_path) as f:
    #     config = yaml.load(f)
    # # convert to dict
    # args = EasyDict(config)
    main(args)

