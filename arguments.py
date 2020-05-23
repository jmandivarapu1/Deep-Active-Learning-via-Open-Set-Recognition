import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',default=True, help='If training is to be done on a GPU')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Name of the dataset used.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size used for training and testing')
    parser.add_argument('--train_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--latent_dim', type=int, default=25088, help='The dimensionality of the VAE latent dimension')
    parser.add_argument('--data_path', type=str, default=' /mnt/iscsi/data/Jay/datasets/cifar10/', help='Path to where the data is')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--num_adv_steps', type=int, default=11, help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_ML_steps', type=int, default=10, help='Number of VAE steps taken for every task model step')
    parser.add_argument('--log_interval', type=int, default=500, help='Print at Certain Steps')
    parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')
    parser.add_argument('--out_path', type=str, default='/mnt/iscsi/data/Jay/ActiveLearning', help='Path to where the output log will be')
    parser.add_argument('--log_name', type=str, default='accuracies.log', help='Final performance of the models will be saved with this name')
    parser.add_argument('--sameIndexs', type=str, default=False, help='Everytime It will be the same experiment')
    parser.add_argument('--device', type=str, default='cpu', help='Cuda Devicce')
    parser.add_argument('--num_classes', type=int, default=10, help='Numer of classes in the dataset')
    parser.add_argument('--method', type=str, default=True, help='Helps in picking the sampler')
    parser.add_argument('--sampler', type=str, default='classifierProbability', help='Helps in picking the sampler')
    parser.add_argument('--samplerMethod', type=str, default='classifierProbability,LatentMeanDistance,WiebullOutlierProbs,Entropy', help='Helps in picking the sampler')
    parser.add_argument('--samplePerClass', type=str, default=False, help='Helps in picking either samples per class or overall')

    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('-p', '--patch-size', default=32, type=int, help='patch size for crops (default: 28)')
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout probability. '
                                                                'If 0.0 no dropout is applied (default)')
    parser.add_argument('-wd', '--weight-decay', default=0.0, type=float, help='Weight decay value (default 0.0)')

    # Architecture and weight-init
    parser.add_argument('-a', '--architecture', default='WRN', help='model architecture (default: WRN)')
    parser.add_argument('--joint', default=False, type=bool,
                        help='construct a joint model, i.e. including classifier and decoder')
    parser.add_argument('--weight-init', default='kaiming-normal',
                        help='weight-initialization scheme (default: kaiming-normal)')
    parser.add_argument('--wrn-depth', default=16, type=int,
                        help='amount of layers in the wide residual network (default: 16)')
    parser.add_argument('--wrn-widen-factor', default=10, type=int,
                        help='width factor of the wide residual network (default: 10)')
    parser.add_argument('--wrn-embedding-size', type=int, default=16,
                        help='number of output channels in the first wrn layer if widen factor is not being'
                            'applied to the first layer (default: 16)')

    # Variational parameters
    parser.add_argument('--train-var', default=True, type=bool,
                        help='Construct and train variational architecture (default: False)')
    parser.add_argument('--var-latent-dim', default=60, type=int, help='Dimensionality of latent space')
    parser.add_argument('--var-beta', default=0.1, type=float, help='weight term for KLD loss (default: 0.1)')
    parser.add_argument('--var-samples', default=1, type=int,
                        help='number of samples for the expectation in variational training (default: 1)')
    parser.add_argument('--model-samples', default=1, type=int,
                        help='Number of stochastic forward inference passes to compute for e.g. MC dropout (default: 1)')

    # Training hyper-parameters
    parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('-lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate (default: 1e-3)')
    parser.add_argument('-bn', '--batch-norm', default=1e-5, type=float, help='batch normalization (default 1e-5)')
    parser.add_argument('-pf', '--print-freq', default=100, type=int, help='print frequency (default: 100)')

    # Resuming training
    parser.add_argument('--resume', default='', type=str, help='path to model to load/resume from(default: none). '
                                                            'Also for stand-alone openset outlier evaluation script')

    # Open set standalone script
    parser.add_argument('--openset-datasets', default='cifar10,FashionMNIST,AudioMNIST,KMNIST,CIFAR10,CIFAR100,SVHN',
                        help='name of openset datasets')

    # Open set arguments
    parser.add_argument('--distance-function', default='cosine', help='Openset distance function (default: cosine) '
                                                                    'choice of euclidean|cosine|mix')
    parser.add_argument('-tailsize', '--openset-weibull-tailsize', default=0.05, type=float,
                        help='tailsize in percent of data (float in range [0, 1]. Default: 0.05')
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    
    return args
