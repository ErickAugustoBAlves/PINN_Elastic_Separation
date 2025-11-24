import numpy as np
import torch
import time
#import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

shape = (101, 101)
spacing = (10, 10)

nbl = 0

boundary_condition = 0

L = (shape[0]-1)*spacing[0]

print('The physical domain is:'+str(L)+' m')


extent = [0, (shape[0]-1)*spacing[0]/1e3, (shape[1]-1)*spacing[1]/1e3, 0]



# Importing the wavefield

path = 'Numerical_wavefield/'

###################################################################################################################

# Homogeneous

ux_num = np.fromfile(path + 'ux_'+str(shape[0])+'_num_1km.bin', dtype=np.float32).reshape((shape[0]+2*nbl, shape[1]+2*nbl))

uz_num = np.fromfile(path + 'uz_'+str(shape[0])+'_num_1km.bin', dtype=np.float32).reshape((shape[0]+2*nbl, shape[1]+2*nbl))

# GDM

#ux_num = np.fromfile(path + 'ux_265_gdm_.bin', dtype=np.float32).reshape((shape[0]+2*nbl, shape[1]+2*nbl))

#uz_num = np.fromfile(path + 'uz_265_gdm_.bin', dtype=np.float32).reshape((shape[0]+2*nbl, shape[1]+2*nbl))

##################################################################################################################



ux_num = ux_num.T

uz_num = uz_num.T




# ============================ 
#     network parameters
# ============================ 

sigma = 1

n_fourier_features = 4


alpha = 1e-4

epochs = 12000

# ============================ 
#     Parameters check
# ============================ 

print('device: ', device)
print('_______')
print('shape: ', shape)
print('_______')
print('spacing: ', shape)
print('_______')
print('nbl: ', nbl)

print('_______')
print('sigma: ', sigma)

print('_______')
print('n_fourier_features: ', n_fourier_features)

print('_______')
print('alpha: ', alpha)

print('_______')
print('Total epochs: ', epochs)

