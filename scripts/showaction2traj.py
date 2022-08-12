import torch
import matplotlib.pyplot as plt
import numpy as np

class bicycle_model:
    def __init__(self, dt=0.25, L=2.8):
        self.x = 0
        self.y = 0
        self.yaw = 0
        self.v = 0
        self.dt = dt
        self.L = L
        self.trajectory = []
        
    def step(self, acc, delta):
        self.v = max(0,self.v)
        self.x = self.x + self.v*np.cos(self.yaw)*self.dt
        self.y = self.y + self.v*np.sin(self.yaw)*self.dt
        self.v = self.v + acc*self.dt
        self.yaw = self.yaw + self.v/self.L*delta*self.dt
        self.trajectory.append([self.x, self.y, self.yaw, self.v])
        
    def reset(self):
        self.x = 0
        self.y = 0
        self.yaw = 0
        self.v = 0
        self.trajectory.clear()
    
    def get_trajectory(self):
        return np.array(self.trajectory)
    
rot = lambda x: np.array([[0,1],[-1,0]])@x

data = torch.load('/home/swyoo/usaywook/RL_carla/gail_experts/CarlaCompound/trajs_carlacompound_rb2.pt')

model = bicycle_model(dt=0.25, L=2.8)
model.reset()
e = 0
path_length = 30
plt.rcParams['figure.figsize'] = (12,4)
lengths = data['lengths']
actions = data['actions']
plt.ion()
for e, l in enumerate(lengths):
    for s in range(l):
        plt.subplot(1,3,1)
        plt.cla()
        plt.imshow(data['camera'][e][s].permute(1,2,0))
        
        plt.subplot(1,3,2)
        plt.cla()
        plt.imshow(data['lidar'][e][s].permute(1,2,0))
        model.reset()
        for action in actions[e, s : s + path_length]:
            model.step(*action)
        traj = model.get_trajectory()[:,:2].T
        traj = rot(traj).T 
        traj = traj*256/32 + np.array([32/2, 32/2 + (32/2-12)])*(256/32)                
        plt.scatter(traj[:,0],traj[:,1], s=10, c='cyan')        
        
        plt.subplot(1,3,3)
        plt.cla()
        plt.imshow(data['birdeye'][e][s].permute(1,2,0))
        plt.draw()
        plt.pause(1e-10)
plt.close()