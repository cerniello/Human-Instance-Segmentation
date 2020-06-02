# Package 
from __future__ import division

import os
import argparse
import socket
import timeit
from datetime import datetime
from tensorboardX import SummaryWriter

# PyTorch 
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes (edited functions from OSVOS-Pytorch)
import cv2
from dataloaders import data_loader as db
from dataloaders import custom_transforms as tr
#from util import visualize as viz
import scipy.misc as sm
import networks.vgg_osvos as vo
from layers.osvos_layers import class_balanced_cross_entropy_loss
from dataloaders.helpers import *
#from mypath import Path


def parse_args():
    parser = argparse.ArgumentParser(description='OSVOS running demo')
    parser.add_argument('--seq_name',
                        default="pID3",
                        help="person sequence to be trained and tested on OSVOS")
    parser.add_argument(
                        '--data_folder',
                        default="data",
                        help='where is located the data folder')
    parser.add_argument('--output_folder', 
                        default="results",
                        help='where the results will be located')
    parser.add_argument(
                        '--epochs',
                        default=50,
                        type=int,
                        help='Number of epochs for the online training'+
                             ' (note that the final nEpochs is epochs*nAvegrad)')
    parser.add_argument(
                        '--batch_size',
                        default=1,
                        type=int,
                        help='Batch size')                       
    parser.add_argument(
                        '--models_dir',
                        default="models",
                        help='where the parent model is located')
    parser.add_argument(
                        '--threshold',
                        default=200,
                        type=int,
                        help='threshold for image to binary mask')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

  working_path = os.getcwd() # /content/path/
  
  args = parse_args()

  # sequence name i.e. "pID3"
  seq_name = args.seq_name

  # data dir
  db_root_dir = os.path.join(working_path, args.data_folder)

  # result dirs
  results_dir = os.path.join(working_path, args.output_folder)
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)

  os.makedirs(os.path.join(results_dir, seq_name), exist_ok=True)

  # models dir
  models_dir = os.path.join(working_path, args.models_dir)

  nEpochs = args.epochs

  vis_net = 0  # Visualize the network?
  vis_res = 0  # Visualize the results?
  nAveGrad = 5  # Average the gradient every nAveGrad iterations
  nEpochs = nEpochs * nAveGrad  # Number of epochs for training
  snapshot = nEpochs  # Store a model every snapshot epochs
  parentEpoch = 240

  epochs_snap = [epoch+1 for epoch in range(nEpochs) if epoch % (nEpochs//20) == (nEpochs//20 - 1)]
  print("Online training snapshots every:", epochs_snap)


  # Parameters in p are used for the name of the model
  p = {
      'trainBatch': args.batch_size,  # Number of Images in each mini-batch
      }
  seed = 0

  parentModelName = 'parent'
  # Select which GPU, -1 if CPU
  gpu_id = 0
  device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

  # Network definition
  net = vo.OSVOS(pretrained=0, models_dir=models_dir)
  net.load_state_dict(torch.load(os.path.join(models_dir, parentModelName+'_epoch-'+str(parentEpoch-1)+'.pth'),
                                map_location=lambda storage, loc: storage))

  # Logging into Tensorboard
  log_dir = os.path.join(models_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()+'-'+seq_name)
  writer = SummaryWriter(log_dir=log_dir)


  net.to(device)  # PyTorch 0.4.0 style

  # Visualize the network
  if vis_net:
      x = torch.randn(1, 3, 480, 854)
      x.requires_grad_()
      x = x.to(device)
      y = net.forward(x)
      g = viz.make_dot(y, net.state_dict())
      g.view()

  # Use the following optimizer
  lr = 1e-8
  wd = 0.0002
  optimizer = optim.SGD([
      {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
      {'params': [pr[1] for pr in net.stages.named_parameters() if 'bias' in pr[0]], 'lr': lr * 2},
      {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
      {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': lr*2},
      {'params': [pr[1] for pr in net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0},
      {'params': [pr[1] for pr in net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0},
      {'params': net.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
      {'params': net.fuse.bias, 'lr': 2*lr/100},
      ], lr=lr, momentum=0.9)


  db_train = db.dataloader(train=True, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
  trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=1)

  # Testing dataset and its iterator
  #db_test = db.DAVIS2016(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
  db_test = db.dataloader(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
  testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

  num_img_tr = len(trainloader)
  num_img_ts = len(testloader)
  loss_tr = []
  aveGrad = 0

  print('n. batches per training epoch: {}, batchsize: {}' .format(num_img_tr, args.batch_size))
  print('n. test images: {}'.format(num_img_ts))

  print("Start of Online Training, sequence: " + seq_name)
  start_time = timeit.default_timer()
  # Main Training and Testing Loop
  for epoch in range(0, nEpochs):
      # One training epoch
      running_loss_tr = 0
      np.random.seed(seed + epoch)
      for ii, sample_batched in enumerate(trainloader):

          inputs, gts = sample_batched['image'], sample_batched['gt']

          #print('-----')
          #print(inputs.shape)
          #print(gts.shape)
          #print('-----')

          # Forward-Backward of the mini-batch
          inputs.requires_grad_()
          inputs, gts = inputs.to(device), gts.to(device)

          outputs = net.forward(inputs)

          # Compute the fuse loss
          loss = class_balanced_cross_entropy_loss(outputs[-1], gts, size_average=False)
          running_loss_tr += loss.item()  # PyTorch 0.4.0 style

          # Print stuff
          if epoch % (nEpochs//20) == (nEpochs//20 - 1):
              running_loss_tr /= num_img_tr
              loss_tr.append(running_loss_tr)
              
              print('[Epoch: %d, numImages: %5d]' % (epoch+1, ii + 1))
              print('Loss: %f' % running_loss_tr)
              writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)

          # Backward the averaged gradient
          loss /= nAveGrad
          loss.backward()
          aveGrad += 1

          # Update the weights once in nAveGrad forward passes
          if aveGrad % nAveGrad == 0:
              writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
              optimizer.step()
              optimizer.zero_grad()
              aveGrad = 0

      # Save the model
      if (epoch % snapshot) == snapshot - 1 and epoch != 0:
          torch.save(net.state_dict(), os.path.join(models_dir, seq_name + '_epoch-'+str(epoch) + '.pth'))

  stop_time = timeit.default_timer()
  print('Online training time: ' + str(stop_time - start_time) + 'seconds')
  print('Online training time: ' + str((stop_time - start_time)/60) + ' minutes')



  #### RUNNING OSVOS ON THE WHOLE SEQUENCE ####

# Testing Phase
if vis_res:
    import matplotlib.pyplot as plt
    plt.close("all")
    plt.ion()
    f, ax_arr = plt.subplots(1, 3)

print('Testing Network')
with torch.no_grad():  # PyTorch 0.4.0 style
    # Main Testing Loop
    for ii, sample_batched in enumerate(testloader):

        img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

        # Forward of the mini-batch
        inputs, gts = img.to(device), gt.to(device)

        outputs = net.forward(inputs)

        for jj in range(int(inputs.size()[0])):
            pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)

            # Save the result, attention to the index jj
            
            # THRESHOLD 
            #ret, binary_mask = cv2.threshold(pred, args.threshold ,255,cv2.THRESH_BINARY)

            sm.imsave(os.path.join(results_dir, seq_name, os.path.basename(fname[jj]) + '.png'), pred)
            
            #sm.imsave(os.path.join(results_dir, seq_name, os.path.basename(fname[jj]) + '.png'), binary_mask)
            #cv2.imwrite(os.path.join(results_dir, seq_name, os.path.basename(fname[jj]) + '.png'), binary_mask)

            if vis_res:
                img_ = np.transpose(img.numpy()[jj, :, :, :], (1, 2, 0))
                gt_ = np.transpose(gt.numpy()[jj, :, :, :], (1, 2, 0))
                gt_ = np.squeeze(gt)
                # Plot the particular example
                ax_arr[0].cla()
                ax_arr[1].cla()
                ax_arr[2].cla()
                ax_arr[0].set_title('Input Image')
                ax_arr[1].set_title('Ground Truth')
                ax_arr[2].set_title('Detection')
                ax_arr[0].imshow(im_normalize(img_))
                ax_arr[1].imshow(gt_)
                ax_arr[2].imshow(im_normalize(pred))
                plt.pause(0.001)


writer.close()
