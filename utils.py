import numpy as np
import torch
import parameters




# In this file we define some utils functions that are used in elastic wavefield decomposition



def Save(data, path, name, shape):
    if path is not None:
        if isinstance(data, np.ndarray):
            data.reshape(shape[0]*shape[1]).tofile(path + name)

        else:
            data = np.array(data)
            data = data.tofile(path + name)
    else:
        if isinstance(data, np.ndarray):
            data.reshape(shape[0]*shape[1]).tofile(name)

        else:
            data = np.array(data)
            data = data.tofile(name)


def pde_grid(shape):
 
    Lx = 1.0
    
    Lz = 1.0
    
    X = np.linspace(0, Lx, shape[0])
    Z = np.linspace(0, Lz, shape[1])

   
    X, Z = np.meshgrid(X, Z)
    
    return X, Z

def boundary_grid(shape):

    Lx = 1.0
    
    Lz = 1.0
    
    X = np.linspace(0, Lx, shape[0]+2)
    Z = np.linspace(0, Lz, shape[1]+2)


    
    X_bound, Z_bound = np.meshgrid(X, Z)
    
    return X_bound, Z_bound
    

def boundary_coords(shape):    
    
    X_bound, Z_bound = boundary_grid(shape)
    
    bc_up_x = np.hstack([X_bound[0,:].reshape(-1,1), Z_bound[0,:].reshape(-1, 1)])
    bc_lower_x = np.hstack([X_bound[-1,:].reshape(-1,1), Z_bound[-1,:].reshape(-1, 1)])
    
    
    bc_left_z = np.hstack([X_bound[:,0].reshape(-1,1), Z_bound[:,0].reshape(-1, 1)])
    bc_right_z = np.hstack([X_bound[:,-1].reshape(-1,1), Z_bound[:,-1].reshape(-1, 1)])

 
    
    bc_all = np.vstack([bc_up_x, bc_lower_x, bc_left_z, bc_right_z])

   
    bc_all = np.unique(bc_all, axis=0)

    
    x_bc = to_torch(bc_all[:, 0])
    z_bc = to_torch(bc_all[:, 1])
    
    
    
    
    sol_bc = torch.empty_like(x_bc)
    sol_bc[:] = parameters.boundary_condition
    
    
    return x_bc, z_bc, sol_bc



def to_torch(data):
    data = torch.tensor(data, dtype=torch.float32).view(-1, 1).to(parameters.device)

    
    return data


def pde_coords(X, Z):    
    
    x_pde = to_torch(X.flatten())
    z_pde = to_torch(Z.flatten())
    
    return x_pde, z_pde
    

def direc_derivative(field, h, axis):   
    return np.gradient(field, h, axis = axis, edge_order=2)


def scalar_decomposition(poisson_sol):
    h = parameters.spacing[0]

    up_num = np.array(np.gradient(poisson_sol, h, edge_order=2))

    upx_PINN = up_num[1, :, :]
    upz_PINN = up_num[0, :, :]



    usx_PINN = parameters.ux_num - upx_PINN
    usz_PINN = parameters.uz_num - upz_PINN
    

    return upx_PINN, upz_PINN, usx_PINN, usz_PINN



