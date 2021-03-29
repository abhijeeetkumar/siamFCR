import os
import numpy as np
import glob

from siamfc import TrackerSiamFC

if __name__ == '__main__':
   seq_dir = os.path.expanduser('/home/mdl/amk7371/CSE586/siamFCR/dataset/OTB/Crossing/')
   img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
   anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')

   net_path = 'network/siamfc_pretrained/model.pth'
   tracker = TrackerSiamFC(netpath=net_path)
   tracker.track(img_files, anno[0], visualize=True)
