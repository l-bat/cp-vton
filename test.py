#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint, GMM_A, GMM_B, Regression

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images
import numpy as np


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default = 1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def test_gmm(opt, test_loader, model, board):
    #model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        c_names = inputs['c_name']
        im = inputs['image']#.cuda()
        im_pose = inputs['pose_image']#.cuda()
        im_h = inputs['head']#.cuda()
        shape = inputs['shape']#.cuda()
        agnostic = inputs['agnostic']#.cuda()
        c = inputs['cloth']#.cuda()
        cm = inputs['cloth_mask']#.cuda()
        im_c =  inputs['parse_cloth']#.cuda()
        im_g = inputs['grid_image']#.cuda()
            
        inp1 = agnostic.numpy()
        inp2 = c.numpy()
        import numpy as np
        np.save('inp1_gmm', inp1)
        np.save('inp2_gmm', inp2)
        torch.onnx.export(model, (agnostic, c), 'gmm.onnx')

        theta = model(agnostic, c)
        # grid, theta = model(agnostic, c)

        # np.save('out_gmm', grid)
        np.save('out_gmm_theta', theta.numpy())
        break

        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        save_images(warped_cloth, c_names, warp_cloth_dir) 
        save_images(warped_mask*2-1, c_names, warp_mask_dir) 



        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)
        
def test_gmm_a(opt, test_loader, model):
    model.eval()

    for step, inputs in enumerate(test_loader.data_loader):
        agnostic = inputs['agnostic']
           
        inp1 = agnostic.numpy()
        import numpy as np
        np.save('feature_a', inp1)
        torch.onnx.export(model, agnostic, 'feature_a.onnx')

        theta = model(agnostic)
        np.save('out_feature_a', theta.numpy())
        break


def test_gmm_b(opt, test_loader, model):
    model.eval()

    for step, inputs in enumerate(test_loader.data_loader):
        c = inputs['cloth']

        inp2 = c.numpy()
        
        import numpy as np
        np.save('feature_b', inp2)
        torch.onnx.export(model, c, 'feature_b.onnx')

        theta = model(c)
        np.save('out_feature_b', theta.numpy())
        break


def test_regression(opt, feature_A, feature_B, model):
    feature_A = torch.from_numpy(feature_A)
    feature_B = torch.from_numpy(feature_B)
    feature_A = feature_A.transpose(2, 3)
    feature_A = feature_A.contiguous().view(1, 512, 16 * 12)
    feature_B = feature_B.view(1, 512, 16 * 12)
    feature_B = feature_B.transpose(1, 2)
    feature_mul = torch.bmm(feature_B, feature_A)
    correlation_tensor = feature_mul.view(1, 16, 12, 16 * 12)
    input1 = correlation_tensor.permute(0, 3, 1, 2)

    model.eval()
    theta = model(input1)
    np_inp = input1.numpy()
    np.save('regression', np_inp)
    torch.onnx.export(model, input1, 'regression.onnx')
    np.save('out_regression', theta.numpy())




def test_tom(opt, test_loader, model, board):
    #model.cuda()
    model.eval()
    
    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        im_names = inputs['im_name']
        im = inputs['image']#.cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic']#.cuda()
        c = inputs['cloth']#.cuda()
        cm = inputs['cloth_mask']#.cuda()
        
        outputs = model(torch.cat([agnostic, c],1))

        inp1 = torch.cat([agnostic, c],1).numpy()
        print("inp1", inp1.shape)
        import numpy as np
        np.save('inp_tom', inp1)
        import onnx
        onnx_inp = torch.cat([agnostic, c], 1)
        torch.onnx.export(model, onnx_inp, 'tom.onnx')

        np.save('out_tom', outputs.numpy())

        # from torchsummary import summary
        # summary(model, input_size=inp1.shape[1:])
        break

        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, 2*cm-1, m_composite], 
                   [p_rendered, p_tryon, im]]
            
        save_images(p_tryon, im_names, try_on_dir) 
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)


def main():
    opt = get_opt()
    print(opt)
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, train_loader, model, board)
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    elif opt.stage == 'GMM_A':
        model = GMM_A()
        with torch.no_grad():
            test_gmm_a(opt, train_loader, model)
    elif opt.stage == 'GMM_B':
        model = GMM_B()
        with torch.no_grad():
            test_gmm_b(opt, train_loader, model)
    elif opt.stage == 'Regression':
        model = Regression(opt)
        feature_a = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/out_feature_a.npy')
        feature_b = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/out_feature_b.npy')
        with torch.no_grad():
            test_regression(opt, feature_a, feature_b, model)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
  
    print('Finished test %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
