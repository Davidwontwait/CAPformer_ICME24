import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np
import cv2

def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='./options/train/pretrain.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    seed = 114514
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)

            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt,0)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
        del resume_state
    else:
        current_step = 0
        start_epoch = 0

    # pre val
    psnr = 0
    ssim = 0
    lpips = 0
    pbar = util.ProgressBar(len(val_loader))
    # for val_data in val_loader:
    #     dark_jpeg = val_data['LQs'][0]
    #     dark_og = val_data['GT'][0]
    #     folder = val_data['folder'][0]
    #     idx_d = val_data['idx']
    #     # dark_og = torch.pow(dark_og,0.3)
    #     # dark_jpeg = torch.pow(dark_jpeg,0.3)
    #
    #     dark_jpeg = util.tensor2img(dark_jpeg)
    #     dark_og = util.tensor2img(dark_og)
    #
    #     a = np.mean(dark_jpeg) / np.mean(dark_og)
    #
    #     dark_og = cv2.convertScaleAbs(dark_og,alpha=a,beta=0)
    #
    #     psnr += util.calculate_psnr(dark_og,dark_jpeg)
    #     ssim += util.calculate_ssim(dark_og,dark_jpeg)
    #     lpips += util.calculate_lpips(dark_og,dark_jpeg)
    #     pbar.update('Beginning Test {} - {}'.format(folder, idx_d))
    #
    # psnr /= len(val_loader)
    # ssim /= len(val_loader)
    # lpips /= len(val_loader)
    #
    # logger.info(f'OG Metirc: PSNR: {psnr:.4f} SSIM: {ssim:.4f} LPIPS: {lpips:.4f} ')

    best_psnr = 0.
    best_ssim = 0.
    best_ssim_i = 0
    best_psnr_i = 0

    best_lpips = 1
    best_lpips_i = 0

    best_loss = 1

    best_metric = 0
    best_metric_i = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr_0:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.2e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.2e} '.format(k, v)
                if rank <= 0:
                    logger.info(message)

                l_final = logs['l_final']
                valid = False
                if l_final < best_loss:
                    best_loss = l_final
                    if current_step >= opt['train']['val_freq']:
                        valid = True
                if current_step % opt['train']['val_freq'] == 0:
                    valid = True

                #### validation
                if valid :
                    img_dir = os.path.join(opt['path']['val_images'], f'step_{current_step}')
                    util.mkdir(img_dir)

                    pbar = util.ProgressBar(len(val_loader))
                    psnr_rlt = {}  # with border and center frames
                    psnr_rlt_avg = {}
                    psnr_total_avg = 0.

                    ssim_rlt = {}  # with border and center frames
                    ssim_rlt_avg = {}
                    ssim_total_avg = 0.

                    lpips_rlt = {}  # with border and center frames
                    lpips_rlt_avg = {}
                    lpips_total_avg = 0.

                    imgs = {}

                    for val_data in val_loader:
                        folder = val_data['folder'][0]
                        idx_d = val_data['idx']  # .item()
                        img_name = val_data['img_name'][0]
                        # border = val_data['border'].item()
                        # img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                        if psnr_rlt.get(folder, None) is None:
                            psnr_rlt[folder] = []
                        if ssim_rlt.get(folder, None) is None:
                            ssim_rlt[folder] = []
                        if lpips_rlt.get(folder, None) is None:
                            lpips_rlt[folder] = []

                        model.feed_data(val_data)
                        model.test()
                        visuals = model.get_current_visuals()
                        rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8
                        # med_img = util.tensor2img(visuals['med'])
                        fea_0 = visuals['fea_0']

                        imgs[f'{img_name}'] = rlt_img

                        # for i in range(len(fea_0)):
                        #     img = fea_0[i].detach()[0].float().cpu()
                        #     img = util.tensor2img(img)
                        #
                        #     save_img_path = os.path.join(img_dir,
                        #                                  f'{img_name}_attnmap_{i}.png')
                        #     if i==0:
                        #         save_img_path = os.path.join(img_dir,
                        #                                  f'{img_name}_MASK.png')
                        #     util.save_fea_map(img, rlt_img, save_img_path)
                        #
                        # save_img_path = os.path.join(img_dir,
                        #                              f'{img_name}.png')
                        # util.save_img(rlt_img, save_img_path)

                        # save_img_path = os.path.join(img_dir,
                        #                              f'{img_name}_GT.png')
                        # util.save_img(gt_img, save_img_path)


                        # calculate PSNR
                        psnr1 = util.calculate_psnr(rlt_img, gt_img)
                        ssim1 = util.calculate_ssim(rlt_img, gt_img)
                        lpips1 = util.calculate_lpips(rlt_img,gt_img)
                        psnr_rlt[folder].append(psnr1)
                        ssim_rlt[folder].append(ssim1)
                        lpips_rlt[folder].append(lpips1)

                        pbar.update('Test {} - {}'.format(folder, idx_d))
                    for k, v in psnr_rlt.items():
                        psnr_rlt_avg[k] = sum(v) / len(v)
                        psnr_total_avg += psnr_rlt_avg[k]
                    for k, v in ssim_rlt.items():
                        ssim_rlt_avg[k] = sum(v) / len(v)
                        ssim_total_avg += ssim_rlt_avg[k]
                    for k, v in lpips_rlt.items():
                        lpips_rlt_avg[k] = sum(v) / len(v)
                        lpips_total_avg += lpips_rlt_avg[k]
                    psnr_total_avg /= len(psnr_rlt)
                    ssim_total_avg /= len(ssim_rlt)
                    lpips_total_avg /= len(lpips_rlt)

                    metric = psnr_total_avg / 45 + ssim_total_avg + 1 - lpips_total_avg

                    logger.info(f'OG Metirc: PSNR: {psnr:.4f} SSIM: {ssim:.4f} LPIPS: {lpips:.4f} ')

                    log_s = '# Current Iter: {} Validation # PSNR: {:.4f}: SSIM: {:.4f} LPIPS:{:.4f} , ' \
                            'MeT:{:.4f}'.format(current_step,psnr_total_avg,ssim_total_avg,lpips_total_avg,metric)
                    logger.info(log_s)
                    if psnr_total_avg > best_psnr:
                        best_psnr = psnr_total_avg
                        best_psnr_i = current_step
                        model.save('Best_PSNR')
                        for i_name,img in imgs.items():
                            save_img_path = os.path.join(opt['path']['val_images'], 'Best_PSNR',)
                            util.mkdir(save_img_path)
                            save_img_path = os.path.join(save_img_path,f'{i_name}.png')
                            util.save_img(img, save_img_path)
                    if ssim_total_avg > best_ssim:
                        best_ssim = ssim_total_avg
                        best_ssim_i = current_step
                        model.save('Best_SSIM')
                        for i_name,img in imgs.items():
                            save_img_path = os.path.join(opt['path']['val_images'], 'Best_SSIM',)
                            util.mkdir(save_img_path)
                            save_img_path = os.path.join(save_img_path,f'{i_name}.png')
                            util.save_img(img, save_img_path)
                    if lpips_total_avg < best_lpips:
                        best_lpips = lpips_total_avg
                        best_lpips_i = current_step
                        model.save('Best_LPIPS')
                        for i_name,img in imgs.items():
                            save_img_path = os.path.join(opt['path']['val_images'], 'Best_LPIPS',)
                            util.mkdir(save_img_path)
                            save_img_path = os.path.join(save_img_path,f'{i_name}.png')
                            util.save_img(img, save_img_path)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_i = current_step
                        model.save('Best_MeT')
                        for i_name,img in imgs.items():
                            save_img_path = os.path.join(opt['path']['val_images'], 'Best_MET',)
                            util.mkdir(save_img_path)
                            save_img_path = os.path.join(save_img_path,f'{i_name}.png')
                            util.save_img(img, save_img_path)

                    log_s = f'Best Psnr:{best_psnr:.4f} at iter {best_psnr_i} , ' \
                            f'Ssim: {best_ssim:.4f} at iter {best_ssim_i}, ' \
                            f'lpips: {best_lpips:.4f} at {best_lpips_i}, ' \
                            f'Met:{best_metric:.4f} at {best_metric_i}'
                    logger.info(log_s)

            #### save models and training states
            # if current_step % opt['logger']['save_checkpoint_freq'] == 0:
            #     if rank <= 0:
            #         logger.info('Saving models and training states.')
            #         model.save(current_step)
            #         model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        # tb_logger.close()


if __name__ == '__main__':
    main()
