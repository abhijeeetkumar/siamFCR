from ParticleFilter import ParticleFilter
from got10k.experiments import *
import glob
import numpy as np

def main():
    #Parameters/Experiment
    particles_num = 40
    experiments = [
        ExperimentVOT('data/OTB_2013', version=2013),
    ]
    #Init
    #img_path = r'./debug/'#'/home/mdl/azk6085/CSE586/siamFCR/Final_Demo/Boy/img/' #r'./data/OTB/boy/'
    img_path = '/home/mdl/azk6085/CSE586/siamFCR/Final_Demo/Boy/img/' #r'./data/OTB/boy/'
    out_path = r'./output'
    #Exec
    run_particle_filter(particles_num,img_path,out_path)

def run_particle_filter(particles_num,img_path,out_path):
  PF=ParticleFilter(particles_num,img_path,out_path)  #Pass Init state
  num_img = sorted(glob.glob(img_path + '*.jpg'))
  print(" PF Img " + str(len(PF.imgs)))
  print(" Actual Img " + str(len(num_img)))
  anno = np.loadtxt(img_path + 'groundtruth_rect.txt', delimiter=',', usecols=range(4))
  #x_0  = np.loadtxt(anno[0], delimiter=' ', usecols=range(4))
  print(anno[0])
  #init = str(anno[0])
  #init = init[0].split(" ")
  #print(init)  
  print(PF.state.output)
  idx=0  
  while PF.img_index<len(PF.imgs):
        print(' idx: '+str(PF.img_index)+' imgs: '+str(len(PF.imgs))) 
        PF.select()
        PF.propagate()
        PF.observe()
        PF.estimate()

if __name__=='__main__':
    main()
