import cv2 as cv
import glob
import numpy as np
from .utils import *
import os
import pdb
import math

class state():
    def __init__(self,x,y,x_dot,y_dot,h_x,h_y,a_dot):
        self.x=x
        self.y=y
        self.x_dot=x_dot
        self.y_dot=y_dot
        self.h_x=h_x
        self.h_y=h_y
        self.a_dot=a_dot
    def draw_dot(self,img,path):
        cv.circle(img, center=(self.x, self.y), radius=1, color=(0, 0, 255), thickness=4)
        self.img=img
        cv.imwrite(path, self.img)
    def draw_rectangle(self,img,path):
        cv.rectangle(img, (self.x - self.h_x,self.y - self.h_y), ( self.x + self.h_x,self.y + self.h_y), (0, 0, 255),thickness=1)
        self.img=img
        cv.imwrite(path, self.img)
    def output(self):
        print('x: %s  y: %s  h_x: %s  h_y: %s  x_dot: %s  y_dot: %s  a_dot: %s'%(self.x,self.y,self.h_x,self.h_y,self.x_dot,self.y_dot,self.a_dot))


class hist():
    def __init__(self,num=8,max_range=360.):  
        self.num=num                          
        self.max_range=max_range
        self.divide=[max_range/num*i for i in range(num)]
        self.height=np.array([0. for i in range(num)])

    def get_hist_id(self,x):
        for i in range(self.num-1):
            if x>=self.divide[i] and x<self.divide[i+1]:
                return i
            elif x>self.divide[-1] and x<=self.max_range:
                return self.num-1
            else:
                return self.num-1

    def update(self,i):
        self.height[i]+=1


class ParticleFilter():
    def __init__(self,particles_num=50):
        self.particles_num=particles_num
        self.DELTA_T=0.05
        self.VELOCITY_DISTURB=4.
        self.SCALE_DISTURB=0.0
        self.SCALE_CHANGE_D=0.001
        self.initial_state=state(x=0,y=0,x_dot=0.,y_dot=0.,h_x=25,h_y=40,a_dot=0.) 
        self.state=self.initial_state
        self.particles=[]
        self.weights = [1. / particles_num] * particles_num  
        self.q = [hist(num=2,max_range=180),hist(num=2,max_range=255),hist(num=10,max_range=255)]

    def init(self, image, initial_state):
        self.state = initial_state
        random_nums=np.random.normal(0,0.4,(self.particles_num,7)) 

        for i in range(self.particles_num):
            x0 = int(initial_state.x + random_nums.item(i, 0) * initial_state.h_x)
            y0 = int(initial_state.y + random_nums.item(i, 1) * initial_state.h_y)
            x_dot0 = initial_state.x_dot + random_nums.item(i, 2) * self.VELOCITY_DISTURB
            y_dot0 = initial_state.y_dot + random_nums.item(i, 3) * self.VELOCITY_DISTURB
            h_x0 = int(initial_state.h_x + random_nums.item(i, 4) * self.SCALE_DISTURB)
            h_y0 = int(initial_state.h_y + random_nums.item(i, 5) * self.SCALE_DISTURB)
            a_dot0 = initial_state.a_dot + random_nums.item(i, 6) * self.SCALE_CHANGE_D
            particle = state(x0, y0, x_dot0, y_dot0, h_x0, h_y0, a_dot0)
            self.particles.append(particle)

        img_first = image
        img_first = cv.cvtColor(img_first, cv.COLOR_BGR2HSV)

        for hist_c in self.q:
            for u in range(hist_c.num):
                a = np.sqrt(initial_state.h_x**2+initial_state.h_y**2)
                f = 0
                weight = []
                x_bin = []
                for i in range(math.floor(initial_state.x - initial_state.h_x), math.floor(initial_state.x + initial_state.h_x)):
                    for j in range(math.floor(initial_state.y - initial_state.h_y), math.floor(initial_state.y + initial_state.h_y)):
                        x_val = img_first[j][i][self.q.index(hist_c)]
                        temp = k(np.linalg.norm((j - initial_state.y, i - initial_state.x)) / a)
                        f += temp
                        weight.append(temp)
                        #pdb.set_trace()
                        x_bin.append(k_delta(hist_c.get_hist_id(float(x_val)) - u))
                hist_c.height[u] = np.sum(np.array(weight) * np.array(x_bin))/f

    def select(self):
        index=get_random_index(self.weights)
        new_particles=[]
        for i in index:
            new_particles.append(state(self.particles[i].x,self.particles[i].y,self.particles[i].x_dot,self.particles[i].y_dot,self.particles[i].h_x,self.particles[i].h_y,self.particles[i].a_dot))
        self.particles=new_particles

    def propagate(self):
        for particle in self.particles:
            random_nums = np.random.normal(0, 0.4, 7)
            particle.x = int(particle.x+particle.x_dot*self.DELTA_T+random_nums[0]*particle.h_x+0.5)
            particle.y = int(particle.y+particle.y_dot*self.DELTA_T+random_nums[1]*particle.h_y+0.5)
            particle.x_dot = particle.x_dot+random_nums[2]*self.VELOCITY_DISTURB
            particle.y_dot = particle.y_dot+random_nums[3]*self.VELOCITY_DISTURB
            particle.h_x = int(particle.h_x*(particle.a_dot+1)+random_nums[4]*self.SCALE_DISTURB+0.5)
            particle.h_y = int(particle.h_y*(particle.a_dot+1)+random_nums[5]*self.SCALE_DISTURB+0.5)
            particle.a_dot = particle.a_dot+random_nums[6]*self.SCALE_CHANGE_D

    def observe(self, image):
        img=image
        img=cv.cvtColor(img , cv.COLOR_BGR2HSV)
        B=[]
        for i in range(self.particles_num):
            if self.particles[i].x<0 or self.particles[i].x>image.shape[1]-1 or self.particles[i].y<0 or self.particles[i].y>image.shape[0]-1:
                B.append(0)
                continue
            self.p = [hist(num=2, max_range=180), hist(num=2, max_range=255), hist(num=10, max_range=255)]
            for hist_c in self.p:
                for u in range(hist_c.num):
                    a = np.sqrt(self.particles[i].h_x ** 2 + self.particles[i].h_y ** 2)
                    f = 0
                    weight = []
                    x_bin = []
                    for m in range(self.particles[i].x - self.particles[i].h_x, self.particles[i].x + self.particles[i].h_x):
                        for n in range(self.particles[i].y - self.particles[i].h_y, self.particles[i].y + self.particles[i].h_y):
                            if n>=image.shape[0]:
                                n=image.shape[0]-1
                            elif n<0:
                                n=0
                            if m>=image.shape[1]:
                                m = image.shape[1] - 1
                            elif m<0:
                                m=0
                            x_val = img[n][m][self.p.index(hist_c)]
                            temp = k(np.linalg.norm((m - self.particles[i].x, n - self.particles[i].y)) / a)
                            f += temp
                            x_bin.append(k_delta(hist_c.get_hist_id(x_val) - u))
                            weight.append(temp)
                    hist_c.height[u] = np.sum(np.array(weight) * np.array(x_bin))/f
            B_temp=B_coefficient(np.concatenate((self.p[0].height,self.p[1].height,self.p[2].height)),np.concatenate((self.q[0].height,self.q[1].height,self.q[2].height)))
            B.append(B_temp)
        for i in range(self.particles_num):
            self.weights[i]=get_weight(B[i])
        self.weights/=sum(self.weights)
        #for i in range(self.particles_num):
            #print('dot: (%d,%d)  weight: %s'%(self.particles[i].x,self.particles[i].y,self.weights[i]))

    def estimate(self):
        self.state.x = np.sum(np.array([s.x for s in self.particles])*self.weights).astype(int)
        self.state.y = np.sum(np.array([s.y for s in self.particles])*self.weights).astype(int)
        self.state.h_x = np.sum(np.array([s.h_x for s in self.particles])*self.weights).astype(int)
        self.state.h_y = np.sum(np.array([s.h_y for s in self.particles])*self.weights).astype(int)
        self.state.x_dot = np.sum(np.array([s.x_dot for s in self.particles])*self.weights)
        self.state.y_dot = np.sum(np.array([s.y_dot for s in self.particles])*self.weights)
        self.state.a_dot = np.sum(np.array([s.a_dot for s in self.particles])*self.weights)
        #print('img: x: %s  y: %s  h_x: %s  h_y: %s  x_dot: %s  y_dot: %s  a_dot: %s'%(self.state.x,self.state.y,self.state.h_x,self.state.h_y,self.state.x_dot,self.state.y_dot,self.state.a_dot))

    def update_predict(self, image):
        self.select()
        self.propagate()
        self.observe(image)
        self.estimate()

        return self.state
