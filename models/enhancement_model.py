import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from models.base_model import BaseModel
from models.loss import CharbonnierLoss2, VGGLoss , ColorLoss,MS_SSIM_L1_LOSS,PSNRLoss
from models.ours import ours
#from models.ours_og import ours as ours
# from models.archs.retinexformer import RetinexFormer as ours
#from models.archs.ours_transformer import Denoiser as ours
# from models.archs.ours1203 import Denoiser as ours
logger = logging.getLogger('base')

class enhancement_model(BaseModel):
    def __init__(self, opt):
        super(enhancement_model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models

        self.ours = ours(opt['ours_model']).to(self.device)
        if opt['dist']:
            self.ours = DistributedDataParallel(self.ours, device_ids=[torch.cuda.current_device()])
        else:
            self.ours = DataParallel(self.ours)
        path = opt['ours_path']
        self.opt_loss = opt['loss']
        # print network
        self.print_network()
        print(f'Loading models from {path}+++++++++++')
        self.load_network(opt['ours_path'], self.ours, True)
        if self.is_train:
            self.ours.train()
            #### loss
            #### loss
            self.cri_ssim = MS_SSIM_L1_LOSS().to(self.device)
            self.cri_color = ColorLoss().to(self.device)
            self.cri_pix = CharbonnierLoss2().to(self.device)
            self.cri_psnr = PSNRLoss().to(self.device)
            self.cri_vgg = VGGLoss().to(self.device)

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

            self.optimizer_ours = torch.optim.Adam(self.ours.parameters(), lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_ours)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingRestartCyclicLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingRestartCyclicLR(
                            optimizer, train_opt['T_period'], eta_mins=[train_opt['lr_G']*0.5,1e-6],))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        # self.med = data['med'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def loss(self,x1,x2):
        loss = 0.

        if self.opt_loss['cb']:
            l_pix = self.cri_pix(x1, x2)
            loss += l_pix
            self.log_dict['l_pix'] = l_pix.item()
        if self.opt_loss['ssim']:
            l_ssim = self.cri_ssim(x1, x2) * 0.001
            loss += l_ssim
            self.log_dict['l_ssim'] = l_ssim.item()
        if self.opt_loss['color']:
            l_color = self.cri_color(x1, x2)*100
            loss += l_color
            self.log_dict['l_color'] = l_color.item()
        if self.opt_loss['vgg']:
            l_vgg = self.cri_vgg(x1, x2)* 0.02
            loss += l_vgg
            self.log_dict['l_vgg'] = l_vgg.item()
        if self.opt_loss['psnr']:
            l_psnr = self.cri_psnr(x1, x2)
            loss += l_psnr
            self.log_dict['l_psnr'] = l_psnr.item()
        return loss

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_ours.zero_grad()


        self.fake_H = self.ours(self.var_L,)

        l_final = self.loss(self.fake_H,self.real_H)  # + l_vgg  # + l_amp + l_pix_amp # + l_pix_amp +l_pha # + l_vgg + l_amp
        l_final.backward()
        self.log_dict['l_final'] = l_final.item()
        torch.nn.utils.clip_grad_norm_(self.ours.parameters(), 0.01)
        self.optimizer_ours.step()
        self.log_dict['m_threshold'] = self.ours.module.get_threshold()

        # self.log_dict['l_pha'] = l_pha.item()

    def test(self):
        self.ours.eval()
        with torch.no_grad():
            self.fake_H = self.ours(self.var_L,)
            self.fea_map_0 = self.ours.module.get_fea_map()

        self.ours.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()


        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        out_dict['fea_0'] = self.fea_map_0
        # out_dict['mask0'] = self.mask0.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        del self.real_H
        del self.fea_map_0
        # del self.mask0
        del self.var_L
        del self.fake_H
        torch.cuda.empty_cache()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.ours)
        if isinstance(self.ours, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.ours.__class__.__name__,
                                             self.ours.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.ours.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network Ours structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.ours, 'Ours', iter_label)
