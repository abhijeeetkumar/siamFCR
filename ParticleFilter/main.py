from ParticleFilter import ParticleFilter
from got10k.experiments import *

def main():
    #Parameters/Experiment
    particles_num = 40
    experiments = [
        #ExperimentGOT10k('data/GOT-10k', subset='test'),
        #ExperimentOTB('data/OTB', version=2013),
        #ExperimentOTB('data/OTB', version=2015),
        ExperimentVOT('data/vot2018', version=2018),
        #ExperimentDTB70('data/DTB70'),
        #ExperimentTColor128('data/Temple-color-128'),
        #ExperimentUAV123('data/UAV123', version='UAV123'),
        #ExperimentUAV123('data/UAV123', version='UAV20L'),
        #ExperimentNfS('data/nfs', fps=30),
        #ExperimentNfS('data/nfs', fps=240)
    ]
    #Init
    img_path = r'./data/vot2018/crossing/'
    out_path = r'./output'
    PF=ParticleFilter(particles_num,img_path,out_path)
    print(PF.imgs)
    #Exec
    while PF.img_index<len(PF.imgs):
        PF.select()
        PF.propagate()
        PF.observe()
        PF.estimate()

if __name__=='__main__':
    main()
