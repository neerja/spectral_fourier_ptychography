
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import image
import math
import os
from IPython import display # for refreshing plotting

# from IPython import display # for refreshing plotting
plot_flag = True

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
    list_leds = np.random.uniform(low = minval, high = maxval, size = (num_leds,2))
    # append a (0,0) bf led
    list_leds = np.append(list_leds,np.zeros([1,2]), axis = 0 )
    # sort the list of leds
    list_leds = list_leds[np.argsort(np.linalg.norm(list_leds,axis = 1))]
    return list_leds

class FPM_setup():
    def __init__(self, pix_size_camera=None, mag=None, wv=None, na_obj=None,Nx=None,Ny=None):
        """
        Constructor that supports different setups based on whether arguments are provided.
        
        Args:
            pix_size_camera (float): Camera pixel size in microns (optional).
            mag (float): Magnification (optional).
            wv (float): Wavelength in microns (optional).
            na_obj (float): Numerical aperture of the objective (optional).
        """

        # Handle camera pixel size, default to 4 microns if no argument is provided
        if pix_size_camera is None:
            self.pix_size_camera = 4  # micron
        elif isinstance(pix_size_camera, (int, float)):
            self.pix_size_camera = pix_size_camera
        else:
            raise TypeError("pix_size_camera must be a scalar (int or float)")
        
        # Handle magnification, default to 1 if not provided
        if mag is None:
            self.mag = 1  # Default magnification if no argument is passed
        elif isinstance(mag, (int, float)):
            self.mag = mag
        else:
            raise TypeError("mag must be a scalar (int or float)")
        
        self.pix_size_object = self.pix_size_camera / self.mag  # Calculate based on magnification

        # Handle wavelength, default to 500e-3 if not provided
        if wv is None:
            self.wv = torch.Tensor([500e-3])  # micron
        elif np.isscalar(wv):
            self.wv = torch.Tensor([wv])   # Convert scalar to Tensor
        elif isinstance(wv, np.ndarray):
            self.wv =  torch.Tensor(wv)  # Convert NumPy array to Tensor
        elif isinstance(wv, torch.Tensor):
            self.wv = wv
        else:
            raise TypeError("wv must be a scalar, None, or a NumPy ndarray")

        self.Nw = len(self.wv) # Number of wavelengths

        # Handle NA, default to 0.05 if not provided
        if na_obj is None:
            self.na_obj = 0.05  # Default NA
        elif isinstance(na_obj, (int, float)):
            self.na_obj = na_obj
        else:
            raise TypeError("na_obj must be a scalar (int or float)")

        if Nx is None:  # use stock object data if no Nx Ny info is provided
            print('Using stock object data')
            path = '/mnt/neerja-DATA/SpectralFPMData/usafrestarget.jpeg'
            im = torch.from_numpy(image.imread(path))[:,:,0] # pick out the first channel
            im = preprocessObject(im)
            (Ny,Nx) = im.shape
            self.Nx = Nx
            self.Ny = Ny
            self.Npixel = Ny
        else:
            print('Using provided Nx and Ny')
            self.Nx = Nx
            self.Ny = Ny
            self.Npixel = Ny

        # create xy grid for object space
        self.createXYgrid()

        # create pupil stack for setup
        self.createPupilStack()

        # create default uniform spectral object
        self.makeUniformSpectralObject()

        # make a uniform spectral object
    def makeUniformSpectralObject(self,obj=None):
        if obj is None: # use stock object data if no object is provided
            path = '/mnt/neerja-DATA/SpectralFPMData/usafrestarget.jpeg'
            im = torch.from_numpy(image.imread(path))[:,:,0]
        obj = preprocessObject(im)
        # make into spectral object
        spectrum = torch.ones([self.Nw,1,1])  # create a spectrum vector with dimension (Nw, 1,1)
        spectral_obj = obj.unsqueeze(0)*spectrum  # elementwise multiply with the sample after expanding dim0 to be wavelength
        self.objstack = spectral_obj
        return self.objstack

    def createPupilStop(self, wv):
        fPupilEdge = self.na_obj/wv # edge of pupipl is NA/wavelength
        fMax = 1/(2*self.pix_size_object)
        df = 1/(self.Npixel*self.pix_size_object)
        fvec = np.arange(-fMax,fMax,df)
        fxc,fyc = np.meshgrid(fvec,fvec)
        # caclulate radius of f coordinates
        frc = np.sqrt(fxc**2+fyc**2)
        pupil = np.zeros_like(frc) 
        pupil[torch.Tensor(frc)<fPupilEdge] = 1 # make everything inside fPupilEdge transmit 100%
        return torch.Tensor(pupil)
    
    def createPupilStack(self):
        pupilstack = torch.zeros([self.Nw,self.Ny,self.Nx])
        for k in np.arange(self.Nw):
            wv = self.wv[k]
            pupilstack[k,:,:]  = self.createPupilStop(wv)
        self.pupilstack = pupilstack
        return self.pupilstack
            
    # create xy grid for object space (used for illumination field)
    def createXYgrid(self):
        # Create x and y vectors for object space
        xvec = np.arange(-self.pix_size_object * self.Nx / 2, self.pix_size_object * self.Nx / 2, self.pix_size_object)
        yvec = np.arange(-self.pix_size_object * self.Ny / 2, self.pix_size_object * self.Ny / 2, self.pix_size_object)

        # Use meshgrid to create a grid of x and y coordinates
        xx, yy = np.meshgrid(xvec, yvec)

        # Combine xx and yy into a single NumPy array and convert to a tensor
        xygrid = np.array([xx, yy])  # Stack them as arrays
        self.xygrid = torch.tensor(xygrid)  # Convert the NumPy array to a tensor

        return self.xygrid

    # create the illumination field given angle of planewave
    def createIllumField(self, illum_angle, wv):
        rady = illum_angle[0]
        radx = illum_angle[1]
        k0 = 2*math.pi/torch.Tensor(wv) # turn into tensor
        ky = k0*math.sin(rady)
        kx = k0*math.sin(radx)
        field = torch.exp(1j*kx*self.xygrid[1] + 1j*ky*self.xygrid[0])
        return field
    
    # create stack of illumination fields for each wavelength at a fixed angle
    def createFixedAngleIllumStack(self, illum_angle):    
        illumstack = torch.zeros([self.Nw,self.Ny,self.Nx],dtype = torch.complex64)  # use complex dtype
        for k in np.arange(self.Nw):  # iterate over wavelengths
            wv = self.wv[k]
            illumstack[k,:,:] = self.createIllumField(illum_angle,wv)
        self.illumstack = illumstack
        return self.illumstack
    
    # compute the measurement given an object and incident wave field and pupil stop
    def forwardSFPM(self):
        # multiply by the sample
        y = torch.zeros_like(self.objstack)
        for k in torch.arange(self.Nw):
            obj = self.objstack[k]
            pupil = self.pupilstack[k]
            field = self.illumstack[k]
            obj_field = field*obj
            # take the fourier transform and center in pupil plane
            pup_obj = torch.fft.fftshift(torch.fft.fft2(obj_field))*pupil
            # multiply object's FFT with the pupil stop and take ifft to get measurement
            y[k] = torch.abs(torch.fft.ifft2(torch.fft.fftshift(pup_obj)))
        y = torch.sum(y,0)
        return (y, pup_obj)

        # create list of illumination angles (leds) to turn on one at a time
    
    def forwardFPM(self,obj,pupil,field):
        # multiply by the sample
        obj_field = field*obj
        # take the fourier transform and center in pupil plane
        pup_obj = torch.fft.fftshift(torch.fft.fft2(obj_field))*pupil
        # multiply object's FFT with the pupil stop and take ifft to get measurement
        y = torch.abs(torch.fft.ifft2(torch.fft.fftshift(pup_obj)))
        return (y, pup_obj)


def createRandomAngleMeasStack(fpm_setup, list_leds, d=75, led_spacing=5):
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

        # create illumination field stack 
        fpm_setup.createFixedAngleIllumStack(illum_angle)

        # simulate the forward measurement
        (y, pup_obj) = fpm_setup.forwardFPM(fpm_setup.objstack[0],fpm_setup.pupilstack[0],fpm_setup.illumstack[0])
        # (y, pup_obj) = fpm_setup.forwardSFPM()
        
            # plot some example measurements
        if k2<5 and plot_flag:
            plt.figure(figsize=(10,10))
            plt.subplot(1,4,1)
            plt.imshow(torch.log10(torch.abs(pup_obj)),'gray')
            plt.subplot(1,4,2)
            plt.imshow(torch.abs(y),'gray')
        measstack[k2,:,:] = y
    return (measstack, list_leds)

class Reconstruction():
    # create a Reconstruction object based on fpm setup
    def __init__(self, fpm_setup, measstack, list_leds, led_spacing=5, dist=75, device = 'cpu'):
        self.Nw = fpm_setup.Nw
        self.Nx = fpm_setup.Nx
        self.Ny = fpm_setup.Ny
        self.measstack = measstack
        self.list_leds = list_leds
        self.Nleds = len(list_leds)
        self.objest = None
        self.led_spacing = led_spacing
        self.dist = dist
        self.device = use_gpu(device)
        self.fpm_setup = fpm_setup
        self.initRecon()

    # initialize the object estimate
    def initRecon(self, obj=None):
        # initialize the object estimate
        if obj is None:
            init_spectrum = torch.ones([self.Nw,1,1]) 
            self.obj_est = self.measstack[0,:,:].unsqueeze(0).to(self.device)*init_spectrum.to(self.device)
        else:
            self.objest = obj
        self.obj_est.requires_grad = True
        self.losses = []
        return self.objest
    
    def hardthresh(self,x,val):
        return torch.maximum(x,torch.tensor(val))
    
    # set hyperparameters
    def parameters(self, step_size = 1e1, num_iters = 100, loss_type = 'MSE', epochs = 1, opt_type = 'Adam'):
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
        if loss_type == 'MSE':
            self.lossfunc = torch.nn.MSELoss()
        else:
            raise ValueError("Loss type not recognized")
        return self.lossfunc
    
    def set_optimizer(self, opt_type):
        if opt_type == 'Adam':
            self.optimizer = torch.optim.Adam([self.obj_est], lr=self.step_size)  # Adam optimizer
        else:
            raise ValueError("Optimizer type not recognized")
        return self.optimizer

    def train(self):
        # move to gpu
        if self.device.type == 'cuda':
            with torch.no_grad():
                obj_est = self.obj_est.to(self.device)
                self.fpm_setup.pupilstack = self.fpm_setup.pupilstack.to(self.device)
                self.measstack = self.measstack.to(self.device)
                self.fpm_setup.illumstack = self.fpm_setup.illumstack.to(self.device)
                
            self.obj_est.requires_grad = True
        
        # train the obj_est
        for k3 in np.arange(self.epochs): # loop through epochs
            for k2 in np.arange(self.num_meas): # loop through measurements
                                # print(k1,k2)
                # get relevant actual measurement and move to gpu
                meas = self.measstack[k2,:,:]
                # loop through wavelength 
                # compute illumination angle from led indices
                led_ind = self.list_leds[k2]   
                led_pos = (led_ind[0]*self.led_spacing,led_ind[1]*self.led_spacing,self.dist) # units = millimeters, (x,y,z)
                illum_angle = (np.arctan(led_pos[1]/led_pos[2]), np.arctan(led_pos[0]/led_pos[2]))

                # create illumination stack
                self.fpm_setup.createFixedAngleIllumStack(illum_angle)
                self.fpm_setup.illumstack = self.fpm_setup.illumstack.to(self.device)

                for k1 in np.arange(self.num_iters): # loop through iterations

                    # simulate the forward measurement
                    self.fpm_setup.objstack = self.obj_est
                    # (yest, pup_obj) = self.fpm_setup.forwardSFPM()
                    (yest, pup_obj) = self.fpm_setup.forwardFPM(self.obj_est[0],self.fpm_setup.pupilstack[0],self.fpm_setup.illumstack[0])
                    # calculate error, aka loss, and backpropagate
                    error = self.lossfunc(yest,meas)
                    self.losses.append(error.detach().cpu())
                    error.backward()
                    # print(error)

                    # Update the object's reconstruction estimate using the optimizer
                    self.optimizer.step()  # Apply the Adam update to obj_est
                    self.optimizer.zero_grad()  # Clear gradients after each step

                    if k1 == self.num_iters - 1:
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
        ax_obj.cla()  # Clear the axis for updated plot
        obj2d = np.sum(self.obj_est.detach().cpu().numpy(), axis=0)  # Sum over wavelengths
        im_obj = ax_obj.imshow(obj2d, cmap='gray')
        ax_obj.set_title('Object 2D Estimate after Epoch {} Meas {} Iter {}'.format(k3+1, k2+1, k1+1))
        colorbar_obj = fig.colorbar(im_obj, ax=ax_obj, orientation='vertical')  # Add colorbar for obj_est

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

def debug_plot(obj): #obj is torch tensor
    plt.imshow(np.abs(np.sum(obj.illumstack.detach().cpu().numpy(), axis=0)))
    plt.colorbar()
    plt.show()