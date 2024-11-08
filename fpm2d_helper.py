
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import image
import math
import os
from IPython import display # for refreshing plotting

# from IPython import display # for refreshing plotting
plot_flag = True

def debug_plot(obj): #obj is torch tensor
    plt.imshow(np.abs(np.sum(obj.detach().cpu().numpy(), axis=0)))
    plt.colorbar()
    plt.show()

def use_gpu(gpu=2):
    if gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        torch.cuda.set_device(gpu)
        device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu') # uncomment this line to use cpu
    torch.cuda.empty_cache()  #empty any variables in cache to free up space
    print(device)
    return device

def preprocessObject(im):
    (Ny,Nx) = im.shape
    # make image square
    if Ny<Nx:
        im = im[:,1:Ny]
    elif Nx<Ny:
            im = im[1:Nx,:]
    # make even dimensions
    if Ny % 2 != 0:
        im = im[:-1,:]
    if Nx % 2 != 0:
        im = im[:,:-1]

    # normalize to max value is 1
    im = im/torch.amax(im)
    return im

def createlist_led(num_leds=100,minval=-3,maxval=3):
    np.random.seed(0) # for debugging and reproducibility
    list_leds = np.random.uniform(low = minval, high = maxval, size = (num_leds,2))
    # append a (0,0) bf led
    list_leds = np.append(list_leds,np.zeros([1,2]), axis = 0 )
    # sort the list of leds
    list_leds = list_leds[np.argsort(np.linalg.norm(list_leds,axis = 1))]
    return list_leds

def createRandomAngleMeasStack(fpm_setup, list_leds, d=75, led_spacing=5):
    global plot_flag # Declare plot_flag as global

    if list_leds is None:
        raise ValueError("list_leds must be provided")
    # create measurement stack
    num_meas = len(list_leds) 
    measstack = torch.zeros(num_meas,fpm_setup.Ny,fpm_setup.Nx)

    for k2 in np.arange(len(list_leds)):
        # take led indices and calculate angle of incidence
        led_ind = list_leds[k2]  
        led_pos = (led_ind[0]*led_spacing,led_ind[1]*led_spacing,d) # units = millimeters, (y,x,z)
        illum_angle = (np.arctan(led_pos[0]/led_pos[2]), np.arctan(led_pos[1]/led_pos[2])) 
        
        plot_flag = False # turn off plotting for now
        # create illumination field stack 
        fpm_setup.createFixedAngleIllumStack(illum_angle)
        # simulate the forward measurement
        (y, pup_obj) = fpm_setup.forwardFPM()
        plot_flag = True # turn plotting back on

            # plot some example measurements
        if k2<5 and plot_flag:
            plt.figure(figsize=(10,10))
            plt.subplot(1,4,1)
            plt.imshow(torch.log10(torch.abs(pup_obj)))
            plt.subplot(1,4,2)
            plt.imshow(y,'gray')
            plt.subplot(1,4,3)
            plt.imshow(torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(y)))))
        measstack[k2,:,:] = y
    return (measstack, list_leds)

class FPM2D_setup():
    def __init__(self):
        self.Nw = 1
        self.mag = 4 # 4x 
        self.pix_size_camera = 4 #  micron
        self.pix_size_object =self.pix_size_camera/self.mag # micron  
        self.wv = 500e-3 # unit = micron,  wavelength is 500 nm
        self.na_obj = 0.05 # low na objective 

        path = '/mnt/neerja-DATA/SpectralFPMData/usafrestarget.jpeg'
        im = torch.from_numpy(image.imread(path))[:,:,0] # pick out the first channel
        im = preprocessObject(im)
        (self.Ny,self.Nx) = im.shape
        self.Npixel = self.Ny
        self.obj = im

        if plot_flag:
            plt.figure()
            plt.imshow(im,'gray')
            plt.colorbar()
            pup_obj = torch.fft.fftshift(torch.fft.fft2(im))
            plt.figure()
            plt.imshow(torch.log10(torch.abs(pup_obj)),'gray')


    def createPupilStop(self):
        fPupilEdge = self.na_obj/self.wv # edge of pupipl is NA/wavelength
        fMax = 1/(2*self.pix_size_object)
        df = 1/(self.Npixel*self.pix_size_object)
        fvec = np.arange(-fMax,fMax,df)
        fxc,fyc = np.meshgrid(fvec,fvec)
        # caclulate radius of f coordinates
        frc = np.sqrt(fxc**2+fyc**2)
        pupil = np.zeros_like(frc) 
        pupil[frc<fPupilEdge] = 1 # make everything inside fPupilEdge transmit 100%
        self.pupil = torch.Tensor(pupil)
        if plot_flag:
            plt.figure()
            plt.imshow(pupil)
            plt.colorbar()
        return torch.Tensor(pupil)

    def createXYgrid(self):
        xvec = np.arange(-self.pix_size_object*self.Nx/2,self.pix_size_object*self.Nx/2,self.pix_size_object)
        yvec = np.arange(-self.pix_size_object*self.Ny/2,self.pix_size_object*self.Ny/2,self.pix_size_object)
        xygrid = torch.Tensor(np.meshgrid(xvec,yvec))
        self.xygrid = xygrid
        return xygrid

    def createIllumField(self, illum_angle, wv):
        rady = illum_angle[0]
        radx = illum_angle[1]
        k0 = 2*math.pi/wv
        ky = k0*math.sin(rady)
        kx = k0*math.sin(radx)
        field = torch.exp(1j*kx*self.xygrid[1] + 1j*ky*self.xygrid[0])
        self.field = field

        if plot_flag:
            plt.figure()
            plt.imshow(torch.angle(field))
        return field

    def forwardFPM(self, obj=None):
        # multiply by the sample
        if obj is None:
            obj_field = self.field*self.obj
        else:
            obj_field = self.field*obj
        # take the fourier transform and center in pupil plane
        pup_obj = torch.fft.fftshift(torch.fft.fft2(obj_field))*self.pupil
        # multiply object's FFT with the pupil stop and take ifft to get measurement
        y = torch.abs(torch.fft.ifft2(torch.fft.fftshift(pup_obj)))

        if plot_flag:
            plt.figure()
            plt.imshow(torch.abs(y.detach().cpu()),'gray')
            plt.title('Measurement')
            plt.figure()
            plt.imshow(torch.abs(self.obj.detach().cpu()),'gray')
            plt.title('Object')
        return (y, pup_obj)
    
    # create stack of illumination fields for each wavelength at a fixed angle
    def createFixedAngleIllumStack(self, illum_angle):    
        self.field = self.createIllumField(illum_angle,self.wv)
        return self.field

    def to(self, device):
        with torch.no_grad():
            # Move each tensor attribute to the specified device
            self.obj = self.obj.to(device)
            self.field = self.field.to(device)
            self.pupil = self.pupil.to(device)
            self.xygrid = self.xygrid.to(device)

class Reconstruction():
    # create a Reconstruction object based on fpm setup
    def __init__(self, fpm_setup, measstack, list_leds, led_spacing=5, dist=75, device = 'cpu'):
        self.Nw = fpm_setup.Nw
        self.Nx = fpm_setup.Nx
        self.Ny = fpm_setup.Ny
        self.measstack = measstack
        self.list_leds = list_leds
        self.Nleds = len(list_leds)
        self.led_spacing = led_spacing
        self.dist = dist
        self.device = use_gpu(device)
        self.fpm_setup = fpm_setup

        self.objest = self.initRecon()  # initialize the object estimate
        with torch.no_grad():
            self.objest = self.objest.to(self.device) # move to gpu
            self.objest.requires_grad = True # allow gradients to be computed
        self.losses = []

    # initialize the object estimate
    def initRecon(self, obj=None):
        # initialize the object estimate
        if obj is None:
            objest = self.measstack[0,:,:].to(self.device) # use the first measurement to initialize
            objest = objest/ torch.amax(objest) # normalize to max value is 1
        else:
            objest = obj

        return objest
    
    # def hardthresh(self,x,val):
    #     return torch.maximum(x,torch.tensor(val))
    
    # set hyperparameters
    def parameters(self, step_size = 1e-3, num_iters = 100, loss_type = '2-norm', epochs = 1, opt_type = 'Adam'):
        # set hyperparameters

        # check that arguments are of right type
        if not isinstance(step_size, (int, float)):
            raise TypeError("step_size must be a scalar (int or float)")
        if not isinstance(num_iters, int):
            raise TypeError("num_iters must be an integer")
        if not isinstance(loss_type, str):
            raise TypeError("loss_type must be a string")
        if not isinstance(epochs, int):
            raise TypeError("epochs must be an integer")
        if not isinstance(opt_type, str):
            raise TypeError("opt_type must be a string")
        
        self.step_size = step_size
        self.num_iters = num_iters
        self.num_meas = len(self.measstack)
        self.epochs = epochs
        self.losses =[]
        self.set_loss(loss_type)
        self.set_optimizer(opt_type)

    def set_loss(self, loss_type):
        if loss_type == '2-norm':
            self.lossfunc = lambda yest, meas: torch.norm(yest-meas)  # 2-norm loss
        else:
            raise ValueError("Loss type not recognized")
        return self.lossfunc
    
    def set_optimizer(self, opt_type):
        if opt_type == 'Adam':
            self.optimizer = torch.optim.Adam([self.objest], lr=self.step_size)  # Adam optimizer
        else:
            raise ValueError("Optimizer type not recognized")
        return self.optimizer
    

    def train(self):
        global plot_flag # Declare plot_flag as global
        plot_flag = False # don't do extra plotting
        
        # train the objest
        for k3 in np.arange(self.epochs): # loop through epochs
            for k2 in np.arange(self.num_meas): # loop through measurements
                                # print(k1,k2)
                # get relevant actual measurement and move to gpu
                meas = self.measstack[k2,:,:].to(self.device) 
                # loop through wavelength 
                # compute illumination angle from led indices
                led_ind = self.list_leds[k2]   
                led_pos = (led_ind[0]*self.led_spacing,led_ind[1]*self.led_spacing,self.dist) # units = millimeters, (x,y,z)
                illum_angle = (np.arctan(led_pos[0]/led_pos[2]), np.arctan(led_pos[1]/led_pos[2]))

                # create illumination for given measurement
                self.fpm_setup.createFixedAngleIllumStack(illum_angle)
                
                for k1 in np.arange(self.num_iters): # loop through iterations

                    # simulate the forward measurement

                    self.fpm_setup.to(self.device) # move to gpu
                    (yest, pup_obj) = self.fpm_setup.forwardFPM(obj=self.objest)

                    # calculate error, aka loss, and backpropagate
                    error = self.lossfunc(yest,meas)
                    self.losses.append(error.detach().cpu())
                    error.backward()

                    # Update the object's reconstruction estimate using the optimizer
                    self.optimizer.step()  # Apply the Adam update to objest
                    self.optimizer.zero_grad()  # Clear gradients after each step

                # at the end of each measurement's iterations, visualize the recon
                try:
                    print(k3, k2, k1)
                    # Visualize the object estimate and its FFT
                    self.visualize(k1,k2,k3)

                except KeyboardInterrupt:
                    break

    def visualize(self,k1,k2,k3):
        # Recreate the figure with two subplots
        fig, (ax_obj, ax_fft) = plt.subplots(1, 2, figsize=(16, 6))  # Create a figure with two axes

        # Plot the loss over time (assuming this is part of a larger script with loss tracking)
        loss_fig, ax_loss = plt.subplots(figsize=(8, 6))  # Create a new figure for the loss
        ax_loss.cla()  # Clear the axis for updated plot
        ax_loss.semilogy(self.losses)
        ax_loss.set_title('Loss over time')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')

        # Plot the object estimate on the left side
        obj2d = self.objest.detach().cpu().numpy()

        im_obj = ax_obj.imshow(obj2d, cmap='gray')
        ax_obj.set_title('Object 2D Estimate after Epoch {} Meas {} Iter {}'.format(k3+1, k2+1, k1+1))
        colorbar_obj = fig.colorbar(im_obj, ax=ax_obj, orientation='vertical')  # Add colorbar for objest

        # Plot the FFT of the object estimate on the right side
        ax_fft.cla()  # Clear the axis for updated plot
        fftobj2d = np.fft.fftshift(np.fft.fft2(obj2d))
        im_fft = ax_fft.imshow(np.log(np.abs(fftobj2d)), cmap='viridis')  # Plot the magnitude of the FFT
        ax_fft.set_title('FFT of Object 2D Estimate')
        colorbar_fft = fig.colorbar(im_fft, ax=ax_fft, orientation='vertical')  # Add colorbar for FFT

        # Save figure
        fig.savefig('object_estimate_fft.png', bbox_inches='tight')  # Save with tight layout

        # Display the updated loss figure
        display.display(loss_fig)  # Display the updated loss figure
        display.clear_output(wait=True)  # Update display

        # Display the figure
        display.display(fig)  # Update display
        display.clear_output(wait=True)  # Update display
