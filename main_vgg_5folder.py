import os 
import time 
import json 
import argparse
import torch 
import torchvision
import random
import numpy as np 
from data import FaceDataset
from tqdm import tqdm 
from torch import nn
from torch import optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.models.resnet import resnet34
from mean_variance_loss import MeanVarianceLoss
import cv2
from networks import vgg16_bn

LAMBDA_1 = 0.2
LAMBDA_2 = 0.05
# LAMBDA_2 = 0
START_AGE = 0
END_AGE = 61
VALIDATION_RATE= 0.1
TEST_RATE= 0.2

random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)


def make_model(num_classes, load_wiki=None):
    model = vgg16_bn(pretrained=True, num_classes=num_classes, load_wiki=load_wiki)
    return model


def train(train_loader, model, criterion1, criterion2, optimizer, epoch, result_directory):

    model.train()
    running_loss = 0.
    running_mean_loss = 0.
    running_variance_loss = 0.
    running_softmax_loss = 0.
    interval = 50
    for i, sample in enumerate(train_loader):
        images = sample['image'].cuda()
        labels = sample['label'].cuda()
        output = model(images)
        mean_loss, variance_loss = criterion1(output, labels)
        # mean_loss, variance_loss = 0,0
        softmax_loss = criterion2(output, labels)
        loss = mean_loss + variance_loss + softmax_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_mean_loss += mean_loss.data
        running_variance_loss += variance_loss.data
        running_softmax_loss += softmax_loss.data
        if (i + 1) % interval == 0:
            print('[%d, %5d] mean_loss: %.3f, variance_loss: %.3f, softmax_loss: %.3f, loss: %.3f'
                  % (epoch, i, running_mean_loss / interval,
                     running_variance_loss / interval,
                     running_softmax_loss / interval,
                     running_loss / interval), end='\r')
            with open(os.path.join(result_directory, 'log'), 'a') as f:
                f.write('[%d, %5d] mean_loss: %.3f, variance_loss: %.3f, softmax_loss: %.3f, loss: %.3f\n'
                        % (epoch, i, running_mean_loss / interval,
                           running_variance_loss / interval,
                           running_softmax_loss / interval,
                           running_loss / interval))
            running_loss = 0.
            running_mean_loss = 0.
            running_variance_loss = 0.
            running_softmax_loss = 0.


def test(test_loader, model):
    model.cuda()
    model.eval()
    mae = 0.
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            image = sample['image'].cuda()
            label = sample['label'].cuda()
            output = model(image)
            m = nn.Softmax(dim=1)
            output = m(output)
            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)
            mae += np.absolute(pred - sample['label'].cpu().data.numpy())
    return mae / len(test_loader)


def predict(model, image):

    model.eval()
    with torch.no_grad():
        image = image.astype(np.float32) / 255.
        image = np.transpose(image, (2,0,1))
        img = torch.from_numpy(image).cuda()
        output = model(img[None])
        m = nn.Softmax(dim=1)
        output = m(output)
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
        mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
        pred = np.around(mean)[0][0]
    return pred


def get_image_list_Morph(image_directory, test_rate, validation_rate):
    
    data_list = []
    for fn in os.listdir(image_directory):
        filepath = os.path.join(image_directory, fn)  
        data_list.append(filepath)
    num = len(data_list)
    np.random.shuffle(data_list)

    test_list = data_list[:int(num*test_rate)]
    val_list = data_list[int(num*test_rate):int(num*(test_rate+validation_rate))]
    train_list = data_list[int(num*(test_rate+validation_rate)):]
    print(len(test_list), len(val_list), len(train_list))

    return train_list, val_list, test_list


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-i', '--image_directory', type=str)
    parser.add_argument('-ls', '--leave_subject', type=int)
    parser.add_argument('-lr', '--learning_rate', type=float)
    parser.add_argument('-e', '--epoch', type=int, default=0)
    parser.add_argument('-n', '--folder_number', type=int, default=0)
    parser.add_argument('-r', '--resume', type=str, default=None)
    parser.add_argument('-rd', '--result_directory', type=str, default=None)
    parser.add_argument('-pi', '--pred_image', type=str, default=None)
    parser.add_argument('-pm', '--pred_model', type=str, default=None)
    parser.add_argument('-lw', '--load_wiki', type=str, default=None)
    return parser.parse_args()


def main():
    
    args = get_args()
    if args.epoch > 0:
        batch_size = args.batch_size

        folder_num = args.folder_number
        # RS
        total_data_temp = [np.loadtxt('list/morph/rs/%d.txt'%(i+1), dtype=str, delimiter=',') for i in range(5)]
        # SE
        # total_data_temp = [np.loadtxt('list/morph/se/%d.txt'%(i+1), dtype=str, delimiter=',') for i in range(5)]
        total_data = []
        for i in range(5):
            folder_data = []
            for j in range(len(total_data_temp[i])):
                folder_data.append('data/morph/img/'+total_data_temp[i][j])
            total_data.append(folder_data)
        train_filepath_list = []
        for i in range(5):
            if i==folder_num:
                continue
            train_filepath_list.extend(total_data[i])
        test_filepath_list = total_data[folder_num]
        args.result_directory = "results/MORPH/rs-mv/k0_f%d"%(folder_num+1)
        print(args.result_directory)
        
        if args.result_directory is not None:
            if not os.path.exists(args.result_directory):
                os.makedirs(args.result_directory)

        transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomApply(
                [torchvision.transforms.RandomAffine(degrees=10, shear=16),
                 torchvision.transforms.RandomHorizontalFlip(p=1.0),
                ], p=0.5),
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.RandomCrop((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_gen = FaceDataset(train_filepath_list, transforms_train)
        train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        test_gen = FaceDataset(test_filepath_list, transforms)
        test_loader = DataLoader(test_gen, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

        model = make_model(END_AGE - START_AGE + 1, load_wiki=args.load_wiki)
        model.cuda()

        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum=0.9, weight_decay=1e-5)
        criterion1 = MeanVarianceLoss(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE).cuda()
        criterion2 = torch.nn.CrossEntropyLoss().cuda()

        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        best_mae_test = np.inf
        best_mae_epoch = -1
        patiant_epoch = 20
        for epoch in range(args.epoch):
            if epoch == 5:
                for param in model.parameters():
                    param.requires_grad = True
            train(train_loader, model, criterion1, criterion2, optimizer, epoch, args.result_directory)
            mae_test = test(test_loader, model)
            scheduler.step()

            print('epoch: %d, test_mae: %3f' % (epoch, mae_test))
            with open(os.path.join(args.result_directory, 'log'), 'a') as f:
                f.write('epoch: %d, mae_test: %3f\n' % (epoch, mae_test))
            if best_mae_test > mae_test:
                best_mae_test = mae_test
                best_mae_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.result_directory, "model_best_mae"))
            elif epoch - best_mae_epoch > patiant_epoch:
                print("early stop at epoch %d"%epoch)
                with open(os.path.join(args.result_directory, 'log'), 'a') as f:
                    f.write('early stop at epoch %d, best_val_mae: %f\n'%(epoch, best_mae_test))
                break        
            with open(os.path.join(args.result_directory, 'log'), 'a') as f:
                f.write('best_mae_epoch: %d, best_val_mae: %f\n'
                        % (best_mae_epoch, best_mae_test))
            print('best_mae_epoch: %d, best_val_mae: %f'
                  % (best_mae_epoch, best_mae_test))


if __name__ == "__main__":
    main()
