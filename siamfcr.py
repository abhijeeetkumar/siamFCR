mport torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR

from got10k.trackers import Tracker

from siamfc import SiamFC
from ParticleFilter import ParticleFilter 
from DCF_net import DCFNet, CFConfig
import argparse
from DCF_util import crop_chw, rect1_2_cxy_wh, cxy_wh_2_bbox


class SiamFCRTracker(Tracker):
    def __init__(self, net_path=None, **kargs):
        super().__init__(
            name='SiamFC', is_deterministic=True)
        self.cfg = self.parse_args(**kargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        print(self.device)

        # setup model
        self.net = SiamFC()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # setup lr scheduler
        self.lr_scheduler = ExponentialLR(
            self.optimizer, gamma=self.cfg.lr_decay)

######Particle filter######## 
        self.PF =
####Discrimitative hyperparameter######
        self.theta = 0.6
        self.beta1 = 0.25
        self.beta2 = 0.02
        self.paritcle_num = 40

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            # inference parameters
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            'adjust_scale': 0.001,
            # train parameters
            'initial_lr': 0.01,
            'lr_decay': 0.8685113737513527,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def init(self, image, box):
        image = np.asarray(image)

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            pad_color=self.avg_color)

        # exemplar features
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            self.kernel = self.net.feature(exemplar_image)

        ######################################################################
        ## initialization for Correlation Filter part
        DCFparser = argparse.ArgumentParser(description='Test DCFNet on OTB')

        DCFparser.add_argument('--model', metavar='PATH', default= 'network/SiamFCR_pretrained/CF_param.pth')

        DCFargs = DCFparser.parse_args()

        self.config_dcf = CFConfig()
        self.DCFnet = DCFNet(self.config_dcf).to(self.device)
        self.DCFnet.load_param(DCFargs.model)

        self.target_pos_dcf, self.target_sz_dcf = rect1_2_cxy_wh(box_1)
        window_sz_dcf = self.target_sz_dcf * (1 + self.config_dcf.padding)
        bbox_dcf = cxy_wh_2_bbox(self.target_pos_dcf, window_sz_dcf)
        patch_dcf = crop_chw(img, bbox_dcf, self.config_dcf.crop_sz)

        self.min_sz_dcf = np.maximum(self.config_dcf.min_scale_factor * self.target_sz_dcf, 4)
        self.max_sz_dcf = np.minimum(img.shape[:2], self.config_dcf.max_scale_factor * self.target_sz_dcf)

        target_dcf = patch_dcf - self.config_dcf.net_average_image
        self.DCFnet.update(torch.Tensor(np.expand_dims(target_dcf, axis=0)).to(self.device))

        self.patch_crop_dcf = np.zeros((self.config_dcf.num_scale, patch_dcf.shape[0], patch_dcf.shape[1], patch_dcf.shape[2]), np.float32)
        ######################################################################



