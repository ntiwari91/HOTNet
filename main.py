from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as io
import os
import random
import time
import socket

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader

# For calculating Flops
#from model_original_flops_cal import HOTNet
from model import HOTNet

from data import get_patch_training_set, get_test_set
from torch.autograd import Variable
from psnr import MPSNR

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--ChDim', type=int, default=31, help='output channel number')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
parser.add_argument('--nEpochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--save_folder', default='TrainedNet/', help='Directory to keep training outputs.')
parser.add_argument('--outputpath', type=str, default='result/', help='Path to output img')
parser.add_argument('--mode', default='test', help='Train or Test.')
opt = parser.parse_args()

print(opt)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(opt.seed)

print('===> Loading datasets')
train_set = get_patch_training_set(opt.upscale_factor, opt.patch_size)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, pin_memory=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False, pin_memory=True)

print('===> Building model')




model = HOTNet().cuda()
print('# network parameters: {}'.format(sum(param.numel() for param in model.parameters())))
model = torch.nn.DataParallel(model).cuda()


optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer, milestones=[10,30,60,120], gamma=0.5)
nEpochss = opt.nEpochs

if opt.nEpochs != 0:
    #load_dict = torch.load(opt.save_folder+"_best.pth".format(opt.nEpochs))
    load_dict = torch.load(opt.save_folder+"_epoch_{}.pth".format(opt.nEpochs))
    opt.lr = load_dict['lr']
    epoch = load_dict['epoch']
    nEpochss=epoch
    model.load_state_dict(load_dict['param'])
    optimizer.load_state_dict(load_dict['adam'])

criterion = nn.L1Loss()

# Generate a timestamped folder name
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
CURRENT_DATETIME_HOSTNAME = current_time + '_' + socket.gethostname()

# Create the full log directory path
log_dir = os.path.join('./tb_logger/CAFL21_Adative_step-size', 'scale' + str(opt.upscale_factor), CURRENT_DATETIME_HOSTNAME)

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# Initialize TensorBoard logger
tb_logger = SummaryWriter(log_dir=log_dir)
current_step = 0

# Define the log file path inside the same directory
log_file_path = os.path.join(log_dir, 'training_log.txt')

# Function to create a directory if it doesn't exist
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"---  New folder created: {path}  ---")
    else:
        print(f"---  Folder already exists: {path}  ---")

# Ensure required folders exist
mkdir(opt.save_folder)
mkdir(opt.outputpath)

def train(epoch, optimizer, scheduler):
    epoch_loss = 0
    global current_step

    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        # with torch.autograd.set_detect_anomaly(True):
        W, Y, Z, X = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
        optimizer.zero_grad()
        W = Variable(W).float()
        Y = Variable(Y).float()
        Z = Variable(Z).float()
        X = Variable(X).float()
        HX, HY, HZ, listX, listY, listZ = model(W)
        alpha = opt.alpha

        loss = criterion(HX, X) + alpha*criterion(HY, Y) + alpha*criterion(HZ, Z)
        for i in range(len(listX) - 1):
            loss = loss + 0.5 * alpha * criterion(X, listX[i]) + 0.5 * alpha * criterion(Y, listY[i]) + 0.5 * alpha * criterion(Z, listZ[i])
        epoch_loss += loss.item()


        tb_logger.add_scalar('total_loss', loss.item(), current_step)
        #tb_logger.add_scalar('psnr', loss.item(), current_step)
        #tb_logger.add_scalar('ssim', loss.item(), current_step)
        current_step += 1

        loss.backward()
        optimizer.step()
        

        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss / len(training_data_loader)

def test(epoch):
    avg_psnr = 0
    avg_time = 0
    model.eval()
    with torch.no_grad():
        for batch in testing_data_loader:
            W, X = batch[0].cuda(), batch[1].cuda()
            W = Variable(W).float()
            X = Variable(X).float()
            torch.cuda.synchronize()
            start_time = time.time()

            HX, HY, HZ, listX, listY, listZ = model(W)
            torch.cuda.synchronize()
            end_time = time.time()

            X = torch.squeeze(X).permute(1, 2, 0).cpu().numpy()
            HX = torch.squeeze(HX).permute(1, 2, 0).cpu().numpy()
            
            psnr = MPSNR(HX,X)

            im_name = batch[2][0]
            # print(im_name)
            # print(end_time - start_time)
            avg_time += end_time - start_time
            (path, filename) = os.path.split(im_name)
            io.savemat('result/'+ filename, {'HX': HX}) #io.savemat(opt.outputpath + filename, {'HX': HX})
            avg_psnr += psnr
    print('VALIDATION')
    print("===> PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("===> time: {:.4f} s".format(avg_time / len(testing_data_loader)))
    tb_logger.add_scalar('psnr', avg_psnr / len(testing_data_loader), current_step)
    with open(log_file_path, 'a') as log_file:
        log_file.write('\n VALIDATION')
        log_file.write(' Epoch: '+str(epoch))
        log_file.write("\n===> PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
        log_file.write("\n===> time: {:.4f} s".format(avg_time / len(testing_data_loader)))
    #print()
    return avg_psnr / len(testing_data_loader)

def checkpoint(epoch, psnr):
    global best_psnr
    global best_epoch

    if epoch % 5 == 0:
    # For saving only best model configuration #
    #if psnr > best_psnr:
        #best_psnr = psnr
        #best_epoch = epoch
        #model_out_path = opt.save_folder+"_best.pth"
    ############################################
        model_out_path = opt.save_folder+"_epoch_{}.pth".format(epoch)
        save_dict = {
            'lr': optimizer.state_dict()['param_groups'][0]['lr'],
            'param': model.state_dict(),
            'adam': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(save_dict, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
        # print("Best checkpoint saved to {}".format(model_out_path))
        # print()

if opt.mode == 'train':
    for epoch in range(nEpochss + 1, 151):
        avg_loss = train(epoch, optimizer, scheduler)
        checkpoint(epoch,test(epoch))
        scheduler.step()
else:
    test(epoch)
