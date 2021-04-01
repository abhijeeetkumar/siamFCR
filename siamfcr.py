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

def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img

def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale

    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]

        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])

        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])

            ##############################################

            ###############################################
            # img = cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness)
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)

    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img


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
        self.particles_num = 40
        self.PF = ParticleFilter(self.particles_num, data/OTB/boy, output)
####Discrimitative hyperparameter######
        self.theta = 0.6
        self.beta1 = 0.25
        self.beta2 = 0.02
        self.PF_part = 0.75
 
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

    def update(self, image):
        image = np.asarray(image)

        # search images
        instance_images = [self._crop_and_resize(
            image, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            pad_color=self.avg_color) for f in self.scale_factors]
        instance_images = np.stack(instance_images, axis=0)
        instance_images = torch.from_numpy(instance_images).to(
            self.device).permute([0, 3, 1, 2]).float()

        # responses
        with torch.set_grad_enabled(False):
            self.net.eval()
            instances = self.net.feature(instance_images)
            responses = F.conv2d(instances, self.kernel) * 0.001
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            t, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - self.upscale_sz // 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale
##############################################################################
       #combining PF and CF
##############################################################################
            x_siamFC = self.center[1] + 1 - (self.target_sz[1] - 1) / 2
            x_siamFC = self.center[0] + 1 - (self.target_sz[0] - 1) / 2
        #Predict using PF




             x_PF, y_PF
       # Compare the IoU for different method
        # box1 for FC
        box1 = np.array([
            x_siamFC,
            y_siamFC,
            self.target_sz[1], self.target_sz[0]])
        # box2 for FC with PF
        box2 = np.array([
            x_PF,
            y_PF,
            self.target_sz[1], self.target_sz[0]])
        ious_FC_PF = self.rect_iou(box1.T, box2.T)            
        if (ious_FC_PF >= self.theta) | (f <= 15):
            # Using Orignal FC
            x_final = x_siamFC
            y_final = y_siamFC
            self.CF_update_predict(img)
            method = 0
        else:
           #Use Particle to modify the center of siamFC
            x_PF_new = self.PF_part*x_PF+ (1 - self.PF_part)*x_siamFC
            y_PF_new = self.PF_part*y_PF+ (1 - self.PF_part)*y_siamFC
           #CF to mo justify the result and modify which is wrong
            A, B = self.CF_update_predict(img)
            x_dcf = int(A[0] - B[0] / 2)
            y_dcf = int(A[1] - B[1] / 2)
            box_dcf = np.array([x_dcf, y_dcf, int(B[0]), int(B[1])])
            iou_dcf_fc = self.rect_iou(box1.T,box_dcf.T)
            iou_dcf_PF = self.rect_iou(box2.T,box_dcf.T)
            if iou_dcf_fc > iou_dcf_PF:
                if iou_dcf_fc > self.beta1:
                    x_final = x_siamFC
                    y_final = y_siamFC
                    method = 0
                    # print("2")
                else:
                    x_final = x_dcf
                    y_final = y_dcf
                    method = 1
                    # print("1")
            else:
                if iou_dcf_PF > self.beta2:
                    x_final = x_PF_new
                    y_final = y_PF_new
                    method = 1
                    # print("3")
                else:
                    x_final = x_dcf
                    y_final = y_dcf
                    method = 2
                    # print("4")
    #############################################################################
        # return the box come back to self.center and update the data
        self.center[1] = x_final - 1 + (self.target_sz[1] - 1) / 2
        self.center[0] = y_final - 1 + (self.target_sz[0] - 1) / 2
        if method == 1:
            box_final = box_dcf
        else:
            box_final = np.array([
                x_final,
                y_final,
                self.target_sz[1], self.target_sz[0]])

        centers = np.array([[x_final], [y_final]])
        self.PF.update(centers)

        return box_final
#############################################################################################################
    ## Using for CF update predict
    def CF_update_predict(self, img):
        im = img  # img
        for i in range(self.config_dcf.num_scale):  # crop multi-scale search region
            window_sz_dcf = self.target_sz_dcf * (self.config_dcf.scale_factor[i] * (1 + self.config_dcf.padding))
            bbox_dcf = cxy_wh_2_bbox(self.target_pos_dcf, window_sz_dcf)
            self.patch_crop_dcf[i, :] = crop_chw(im, bbox_dcf, self.config_dcf.crop_sz)

        search_dcf = self.patch_crop_dcf - self.config_dcf.net_average_image
        response_dcf = self.DCFnet(torch.Tensor(search_dcf).to(self.device))
        peak_dcf, idx_dcf = torch.max(response_dcf.view(self.config_dcf.num_scale, -1), 1)
        idxcpu = idx_dcf.cpu()
        peakcpu = peak_dcf.data.cpu().numpy() * self.config_dcf.scale_penalties
        best_scale_dcf = np.argmax(peakcpu)
        r_max, c_max = np.unravel_index(idxcpu[best_scale_dcf], self.config_dcf.net_input_size)

        if r_max > self.config_dcf.net_input_size[0] / 2:
            r_max = r_max - self.config_dcf.net_input_size[0]
        if c_max > self.config_dcf.net_input_size[1] / 2:
            c_max = c_max - self.config_dcf.net_input_size[1]
        window_sz_dcf = self.target_sz_dcf * (self.config_dcf.scale_factor[best_scale_dcf] * (1 + self.config_dcf.padding))

        # print(np.array([c_max, r_max]) * window_sz / config.net_input_size)
        self.target_pos_dcf = self.target_pos_dcf + np.array([c_max, r_max]) * window_sz_dcf / self.config_dcf.net_input_size
        self.target_sz_dcf = np.minimum(np.maximum(window_sz_dcf / (1 + self.config_dcf.padding), self.min_sz_dcf), self.max_sz_dcf)

        window_sz_dcf = self.target_sz_dcf * (1 + self.config_dcf.padding)
        bbox_dcf = cxy_wh_2_bbox(self.target_pos_dcf, window_sz_dcf)
        patch_dcf = crop_chw(im, bbox_dcf, self.config_dcf.crop_sz)
        target_dcf = patch_dcf - self.config_dcf.net_average_image
        self.DCFnet.update(torch.Tensor(np.expand_dims(target_dcf, axis=0)).to(self.device), lr=self.config_dcf.interp_factor)

        # print(self.target_pos_dcf)
        # return target_pos_dcf, self.target_sz_dcf
        # return self.target_pos_dcf[0],self.target_pos_dcf[1],self.target_sz_dcf[0],self.target_sz_dcf[1]
        return self.target_pos_dcf, self.target_sz_dcf
####################################################################################################################

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img, f)
            times[f] = time.time() - begin
            # print(boxes[f, :])
            if visualize:
                show_image(img, boxes[f, :])

        return boxes, times


    def step(self, batch, backward=True, update_lr=False):
        if backward:
            self.net.train()
            if update_lr:
                self.lr_scheduler.step()
        else:
            self.net.eval()

        z = batch[0].to(self.device)
        x = batch[1].to(self.device)

        with torch.set_grad_enabled(backward):
            responses = self.net(z, x)
            labels, weights = self._create_labels(responses.size())
            loss = F.binary_cross_entropy_with_logits(
                responses, labels, weight=weights, size_average=True)

            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels, self.weights

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # pos/neg weights
        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        weights = np.zeros_like(labels)
        weights[labels == 1] = 0.5 / pos_num
        weights[labels == 0] = 0.5 / neg_num
        weights *= pos_num + neg_num

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        weights = weights.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))
        weights = np.tile(weights, [n, c, 1, 1])

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        self.weights = torch.from_numpy(weights).to(self.device).float()

        return self.labels, self.weights

########################################################################################################################
    def rect_iou(self, rects1, rects2, bound=None):
        r"""Intersection over union.
        Args:
            rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
                (left, top, width, height).
            rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
                (left, top, width, height).
            bound (numpy.ndarray): A 4 dimensional array, denotes the bound
                (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
        """
        assert rects1.shape == rects2.shape
        if bound is not None:
            # bounded rects1
            rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
            rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
            rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
            rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
            # bounded rects2
            rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
            rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
            rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
            rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

        x1 = np.maximum(rects1[..., 0], rects2[..., 0])
        y1 = np.maximum(rects1[..., 1], rects2[..., 1])
        x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                        rects2[..., 0] + rects2[..., 2])
        y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                        rects2[..., 1] + rects2[..., 3])

        w = np.maximum(x2 - x1, 0)
        h = np.maximum(y2 - y1, 0)

        rects_inter = np.stack([x1, y1, w, h]).T

        areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

        areas1 = np.prod(rects1[..., 2:], axis=-1)
        areas2 = np.prod(rects2[..., 2:], axis=-1)
        areas_union = areas1 + areas2 - areas_inter

        eps = np.finfo(float).eps
        ious = areas_inter / (areas_union + eps)
        ious = np.clip(ious, 0.0, 1.0)

        return ious
