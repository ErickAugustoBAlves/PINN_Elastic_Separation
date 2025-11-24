import numpy as np
import torch
import time
import utils
from my_model import simple_NN
import os
import parameters

 



device = parameters.device

shape = parameters.shape

spacing = parameters.spacing

nbl = parameters.nbl

boundary_condition = parameters.boundary_condition

epochs = parameters.epochs

alpha = parameters.alpha

L = parameters.L

uz_num = parameters.uz_num

ux_num = parameters.ux_num

sigma = parameters.sigma




x_bc, z_bc, sol_bc = utils.boundary_coords(shape)
X, Z = utils.pde_grid(shape)
x_pde, z_pde = utils.pde_coords(X, Z)



def run():

    class PINN_Poisson:
    
    
        def create_model(self, load):
        
            my_model = simple_NN()
            my_model.to(device)

            if load is False:
            
                return my_model
        
            else: 
                path = 'GDM_decomposition/GDM_reduced/Saved_NN/paramets_100FF_.pth'

                checkpoint = torch.load(path)

                my_model.load_state_dict(checkpoint['model_state_dict'])

            
            return my_model

        
        def training(self, epochs, source, x_pde, z_pde, x_bc, z_bc, sol_bc, alpha, ux, uz, load, save, one_sol):

        
        
            Loss_PDE_values = []
            Loss_BC_values = []
        
            Loss_values = []

            lr_values = []
        
        #creating model
            my_model = self.create_model(load)
        
        # Model traning...
        
            if load is True:
                path = 'GDM_decomposition/GDM_reduced/Saved_NN/paramets_100FF_.pth'

                checkpoint = torch.load(path)

                opt = torch.optim.Adam(my_model.parameters(), lr=1e-3)
                opt.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                start_epoch = checkpoint['epoch'] + 1
                Loss_values = checkpoint['loss']
##################################################################################            
            
            if one_sol is True:
            
                t_ini = time.time()
                
                sol_approx = my_model(x_pde, z_pde)
                sol_PINN = sol_approx[:, 0]

                t_end = time.time()
                
                return sol_PINN, t_end - t_ini
              
            
##################################################################################
            else:
                start_epoch = 0
                opt = torch.optim.Adam(my_model.parameters(), lr=1e-3)
                scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
            
            
            
        


            ep_range = np.linspace(start_epoch, start_epoch + epochs, start_epoch + epochs)
        
            x_pde.requires_grad_(True)
            z_pde.requires_grad_(True)
        
        
            if type(source) is not torch.Tensor:

                source = source.reshape(-1, 1)
                source = torch.from_numpy(source).to(dtype=x_pde.dtype, device=x_pde.device)

        
            t_ini = time.time()
            
            for epoch in range(start_epoch, epochs + start_epoch):
            
                my_model.train()
            
                w_guess = my_model(x_pde, z_pde)

                w_guess_bc = my_model(x_bc, z_bc)

                
                grads = torch.autograd.grad(w_guess, [x_pde, z_pde], grad_outputs=torch.ones_like(w_guess), create_graph=True)

                w_guess_x = grads[0]
                w_guess_z = grads[1]

                w_guess_xx = torch.autograd.grad(w_guess_x, x_pde, grad_outputs=torch.ones_like(w_guess_x), create_graph=True)[0]
                w_guess_zz = torch.autograd.grad(w_guess_z, z_pde, grad_outputs=torch.ones_like(w_guess_z), create_graph=True)[0]


                laplacian_w = w_guess_xx + w_guess_zz

                pde_residual = laplacian_w - source
                bc_residual = w_guess_bc - sol_bc
            
            

                Loss_PDE = torch.mean(pde_residual**2)
                Loss_bc =  torch.mean(bc_residual**2)
                
                loss = alpha*Loss_PDE + Loss_bc
    
                opt.zero_grad()
                loss.backward()
                opt.step()

                scheduler.step()
                current_lr = opt.param_groups[0]['lr']
                lr_values.append(current_lr)

   

                Loss_values.append(loss.item())

                Loss_PDE_values.append(Loss_PDE.item())

                Loss_BC_values.append(Loss_bc.item())
              
            
                if save is True:
                    if epoch % 1000 == 0 or epoch == epochs - 1: 
                        print(epoch)
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': my_model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': Loss_values
                        }

                        path = 'GDM_decomposition/GDM_reduced/Saved_NN/paramets_100FF_.pth'

                        torch.save(checkpoint, path)
                
                print(f"Running: {(epoch / epochs) * 100:.2f}%", end='\r', flush=True)


                if epoch % 100 == 0 or epoch == epochs - 1:
                    print("alpha = ", alpha)
                    print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e}")
                    print(f"Epoch {epoch:5d} | Loss_PDE: {Loss_PDE_values[-1]:.6e}")
                    print(f"Epoch {epoch:5d} | Loss_BC: {Loss_BC_values[-1]:.6e}")
                    print('___________________________________________')
                
            
            
            
            

            poisson_sol_pinn = my_model(x_pde, z_pde)

            poisson_sol_pinn = poisson_sol_pinn[:, 0]

            poisson_sol_pinn = poisson_sol_pinn.detach().cpu().numpy().reshape((shape[0]+2*nbl,shape[1]+2*nbl))
        
            t_end = time.time()
            
            return poisson_sol_pinn, t_end - t_ini, Loss_values



####### PINN ########
    h = spacing[0]

    div_u = utils.direc_derivative(uz_num, h, axis = 0) + utils.direc_derivative(ux_num, h, axis = 1)

    source = div_u

    true_source = L**2*source 
    
    PINN = PINN_Poisson()
    
    poisson_sol, t, Loss_values = PINN.training(epochs, true_source, x_pde, z_pde, x_bc, z_bc, sol_bc, alpha, ux_num, uz_num, load = False, save = False, one_sol = False)


# ================= SCALAR FORMULATION ========================


    upx_PINN, upz_PINN, usx_PINN, usz_PINN = utils.scalar_decomposition(poisson_sol)



    path = 'Wavefield_decomposition/'#sig_1_alpha_1e-4/10Hz/Activation_function/'
    os.makedirs(path, exist_ok=True)
 
    utils.Save(upx_PINN, path, 'upx_shape_'+str(shape[0])+'_alpha_'+str(alpha)+'_epochs_'+str(epochs)+'_256_10Hz_sig_1.bin', shape)
    utils.Save(upz_PINN, path, 'upz_shape_'+str(shape[0])+'_alpha_'+str(alpha)+'_epochs_'+str(epochs)+'_256_10Hz_sig_1.bin', shape)

    utils.Save(usx_PINN, path, 'usx_shape_'+str(shape[0])+'_alpha_'+str(alpha)+'_epochs_'+str(epochs)+'_256_10Hz_sig_1.bin', shape)
    utils.Save(usz_PINN,  path, 'usz_shape_'+str(shape[0])+'_alpha_'+str(alpha)+'_epochs_'+str(epochs)+'_256_10Hz_sig_1.bin', shape)
        
    utils.Save(Loss_values, path, 'Loss_values_'+str(shape[0])+'_alpha_'+str(alpha)+'_epochs_'+str(epochs)+'_256_10Hz_sig_1.bin', shape)

    utils.Save(t, path, 'Time_'+str(shape[0])+'_alpha_'+str(alpha)+'_epochs_'+str(epochs)+'_256_10Hz_sig_1.bin',shape)

  
    print('Files Saved: Homogeneous model, 10Hz, sigma = '+str(sigma)+' and alpha = '+str(alpha)+'')






    




