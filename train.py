import argparse, os, glob
from pickle import TRUE
import torch,pdb
import math, random, time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from model.model import _NetG,_NetD,_NetD_512
from model.model import _NetG,_NetD,_NetD_512
#from dataset_dep import DatasetFromHdf5
from torchvision.utils import save_image
from random import randint, seed
import random
import cv2
from dataloader.EyeQ_sample import EyeQ_Dataset
import albumentations as A 
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from pytorch_msssim import ssim,ms_ssim


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100  
    return 20 * math.log10(1.0 / rmse)

# Training settings
parser = argparse.ArgumentParser(description="PyTorch OTE-GAN") 
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=150, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default="/home/vvasa1/GSL research/EyeQ/EyeQ_preprocess/OTE-GAN/OTE-GAN/Experiement/Exp-1/checkpoint/model_denoise_49_45.pth", type=str, help="Path to resume model (default: none")
parser.add_argument("--start-epoch", default=50, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, (default: 1)")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--noise_sigma", default=70, type=int, help="standard deviation of the Gaussian noise (default: 50)")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--sigma", default=45, type=int)
parser.add_argument("--num_rand",default=[1000,1000,1000,1000,1000,1000], type=list)
parser.add_argument("--root", default="/home/vvasa1/GSL research/EyeQ/EyeQ_preprocess/original_crop", type=str)
parser.add_argument("--file_dir",default="/home/vvasa1/GSL research/EyeQ/data/Label_EyeQ_train.csv", type=str)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num

def main():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cuda = opt.cuda
    # if cuda: 
    #     print("=> use gpu id: '{}'".format(opt.gpus))
    #     os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    #     if not torch.cuda.is_available():
    #             raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    # data_list = glob.glob(opt.trainset+"*.h5")
    num_random = opt.num_rand
    print("===> Building model")
    model = _NetG()
    discr = _NetD()


    #
    # a,b=get_parameter_number(model)
    # print(model)
    # print(a,b)
    print("===> Setting GPU")
    if cuda:
        #model = model.cuda()
        #discr = discr.cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        #  dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
            discr = nn.DataParallel(discr)

        model.to(device=device)
        discr.to(device=device)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
            discr.load_state_dict(checkpoint["discr"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict(),strict=True)
            discr.load_state_dict(weights['discr'].state_dict(),strict=True)
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    data_transforms = {
        # 'HQ': 
        # A.Compose([
        # A.Resize(height=512,width=512),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.Normalize(mean=0.5, std=1.0),
        # ToTensorV2(),
        # ]),

        # 'LQ': T.Compose([
        #     A.Resize(height=512,width=512),
        #     A.HorizontalFlip(p=0.5),
        #     A.VerticalFlip(p=0.5),
        #     A.Normalize(mean=0.5, std=1.0),
        #     ToTensorV2(),  
        # ]),

        'HQ': T.Compose([
                T.Resize((256,256)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                #T.RandomRotation((-180,180)),
                T.ToTensor()
        ]),
        'LQ': T.Compose([
                T.Resize((256,256)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                #T.RandomRotation((-180,180)),
                T.ToTensor()
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("===> Setting Optimizer")
    G_optimizer = optim.RMSprop(model.parameters(), lr=opt.lr/2)
    D_optimizer = optim.RMSprop(discr.parameters(), lr=opt.lr)

    print("===> Training")
    OT_CONSTRAIN =[]
    GLOSS=[]
    Psnr = []
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        ot_loss = 0
        Gloss=0
        num_rand = 0
        for i in num_random:
            train_set = EyeQ_Dataset(root=opt.root,file_dir=opt.file_dir,select_number=i,transform_HQ=data_transforms['HQ'],transform_PQ=data_transforms['LQ'])
           
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
                batch_size=opt.batchSize,shuffle=True)
            a,b,d=train(training_data_loader, G_optimizer, D_optimizer, model, discr, epoch,num_rand)
            ot_loss += a
            Gloss+=b
            num_rand += 1
        ot_loss = ot_loss / len(num_random)
        Gloss = Gloss / len(num_random)

        OT_CONSTRAIN.append(format(ot_loss))
        GLOSS.append(format(Gloss))
        Psnr.append(format(d))
        save_checkpoint(model, discr, epoch)

        print(ot_loss)

    file = open('OTE-GAN/Experiment/exp11/checksample/mse_'+str(opt.nEpochs)+'_'+str(opt.sigma)+'.txt','w')
    for mse in OT_CONSTRAIN:
        file.write(mse+'\n')
    file.close()

    file = open('OTE-GAN/Experiment/exp11/checksample/Gloss_'+str(opt.nEpochs)+'_'+str(opt.sigma)+'.txt', 'w')
    for g in GLOSS:
        file.write(g + '\n')
    file.close()

    file = open('OTE-GAN/Experiment/exp11/checksample/PSNR_'+str(opt.nEpochs)+'_'+str(opt.sigma)+'.txt', 'w')
    for g in Psnr:
        file.write(g + '\n')
    file.close()
    # psnr = eval_dep(model)
    # print("Final psnr is:",psnr)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def train(training_data_loader, G_optimizer, D_optimizer, model, discr, epoch,num_rand):

    lr = adjust_learning_rate(D_optimizer, epoch-1)
    ot_loss = []
    Gloss=[]
    Dloss = []
    Psnr = []
    Gidentity = []
    for param_group in G_optimizer.param_groups:
        param_group["lr"] = lr/2
    for param_group in D_optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, D_optimizer.param_groups[0]["lr"]))
    #model.train()
    #discr.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        
        target = Variable(batch[1])
        raw = Variable(batch[0])
        #print(target)
        #print('hq',batch[3])
        #print('pq',batch[2])

        if opt.cuda:
            target = target.cuda()
            raw = raw.cuda()
            #noise=noise.cuda()
            input = raw


        # train discriminator D
        discr.zero_grad()
        #print(target)
        D_result = discr(target).squeeze()
        D_real_loss = -D_result.mean()

        G_result = model(input)
        D_result = discr(G_result.data).squeeze()

        D_fake_loss = D_result.mean()

        D_train_loss = D_real_loss + D_fake_loss
        Dloss.append(D_train_loss.data)

        D_train_loss.backward()
        D_optimizer.step()

        #gradient penalty
        discr.zero_grad()
        alpha = torch.rand(target.size(0), 1, 1, 1)
        alpha1 = alpha.cuda().expand_as(target)
        interpolated1 = Variable(alpha1 * target.data + (1 - alpha1) * G_result.data, requires_grad=True)
        
        out = discr(interpolated1).squeeze()

        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated1,
                                   grad_outputs=torch.ones(out.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        gp_loss = 10 * d_loss_gp

        gp_loss.backward()
        D_optimizer.step()

        # train generator G
        discr.zero_grad()
        model.zero_grad()

        G_result = model(input)
        D_result = discr(G_result).squeeze()

        #print(torch.max(G_result))
        #print(torch.max(input))
        constrain = 1 - ms_ssim(G_result,input,data_range=1.0,size_average=True)
        # mse_loss = (torch.mean((G_result- input)**2))**0.5
        ot_loss.append(constrain.data)
        

        #G_train_loss = - D_result.mean() + opt.sigma * mse_loss + 10 * G_identity + 20*G_l
        G_train_loss = - D_result.mean() + opt.sigma * constrain 
        Gloss.append(G_train_loss)
        G_train_loss.backward()
        G_optimizer.step()
        pp=PSNR(input,G_result)
        Psnr.append(pp)

        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss_G: {:.5}, Loss_constrain: {:.5}".format(epoch, iteration, len(training_data_loader), G_train_loss.data, constrain.data))
    save_image(G_result.data, '/home/vvasa1/GSL research/EyeQ/EyeQ_preprocess/OTE-GAN/OTE-GAN/Experiement/Exp-1/'+str(epoch)+'_'+str(num_rand)+'_'+'output.png')
    save_image(input.data, '/home/vvasa1/GSL research/EyeQ/EyeQ_preprocess/OTE-GAN/OTE-GAN/Experiement/Exp-1/'+str(epoch)+'_'+str(num_rand)+'_'+'input.png')
    save_image(target.data, '/home/vvasa1/GSL research/EyeQ/EyeQ_preprocess/OTE-GAN/OTE-GAN/Experiement/Exp-1/'+str(epoch)+'_'+str(num_rand)+'_'+'gt.png')


    return torch.mean(torch.FloatTensor(ot_loss)),torch.mean(torch.FloatTensor(Gloss)),torch.mean(torch.FloatTensor(Psnr))
   
def save_checkpoint(model, discr, epoch):
    model_out_path = "/home/vvasa1/GSL research/EyeQ/EyeQ_preprocess/OTE-GAN/OTE-GAN/Experiement/Exp-1/checkpoint/" + "model_denoise_"+str(epoch)+"_"+str(opt.sigma)+".pth"
    state = {"epoch": epoch ,"model": model, "discr": discr}
    if not os.path.exists("/home/vvasa1/GSL research/EyeQ/EyeQ_preprocess/OTE-GAN/OTE-GAN/Experiement/Exp-1/checkpoint/"):
        os.makedirs("/home/vvasa1/GSL research/EyeQ/EyeQ_preprocess/OTE-GAN/OTE-GAN/Experiement/Exp-1/checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()