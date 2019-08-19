import argparse
import os
import random
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-dp','--data_path', default='./test_image/', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-g','--gpu', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('-smp','--saved_model_path',default='./models_resnet/',type=str, help='None')
best_acc1 = 0

args = parser.parse_args()
ngpu_per_node = len(args.gpu.split(','))

def get_number_of_classification(filepath):
    for path, dirnames, _ in os.walk(filepath):
        break
    for dirname in dirnames:
        if '.ipynb' in dirname:
            dirnames.remove(dirname)
    dirnames.sort()
#     print('nihao')
#     print(dirnames, len(dirnames))
    return path, dirnames, len(dirnames)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

# _,_,num_classes = get_number_of_classification('./dataset/train/')
# print(num_classes)

classes, class_to_idx = find_classes('./dataset/train')
num_classes = len(classes)

def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = len(args.gpu.split(','))
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    # global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))

    model = models.__dict__[args.arch](num_classes = num_classes )

    if ngpus_per_node == 1:
        torch.cuda.set_device(int(args.gpu))
        model = model.cuda(int(args.gpu))
    #multi-gpu
    elif ngpus_per_node > 1:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(torch.load(args.saved_model_path + 'resnet50_checkpoint.pth.tar')['state_dict'])

    cudnn.benchmark = True

    # Data loading code
    testdir = args.data_path
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,])

    torch.no_grad()

    for image_path in os.listdir(testdir):
        image = Image.open(testdir + '/' + image_path)
        image = transform(image).unsqueeze(0)
        print(image.size())
        output = model(image)
        print(output)
        print(output.size())
        _,predicted = torch.max(output,1)
        print(image_path, predicted)




if __name__ == '__main__':
    main()
