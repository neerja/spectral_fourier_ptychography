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
    np.random.seed(0) # for debugging and reproducibility
    list_leds = np.random.uniform(low = minval, high = maxval, size = (num_leds,2))
    # append a (0,0) bf led
    list_leds = np.append(list_leds,np.zeros([1,2]), axis = 0 )
    # sort the list of leds
    list_leds = list_leds[np.argsort(np.linalg.norm(list_leds,axis = 1))]
    return list_leds

class FPM_setup():
    def __init__(self, pix_size_camera=None, mag=None, wv=None, na_obj=None,Nx=None,Ny=None, led_spacing=5, dist=75, list_leds=None):
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
            raise TypeError("wv must be a scalar, None, or a NumPy array or torch Tensor")

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

        # Initialize LED-related attributes
        self.led_spacing = led_spacing
        self.dist = dist
        self.list_leds = list_leds
    
    def __str__(self):
        # Return a string representation of the FPM_setup object
        return (
            f"FPM Setup Parameters:\n"
            f"--------------------\n"
            f"Camera pixel size: {self.pix_size_camera} microns\n"
            f"Magnification: {self.mag}x\n"
            f"Object pixel size: {self.pix_size_object:.3f} microns\n"
            f"Numerical aperture: {self.na_obj}\n"
            f"Wavelength(s): {self.wv.numpy()} microns\n"
            f"Number of wavelengths: {self.Nw}\n"
            f"Image dimensions: {self.Nx} x {self.Ny} pixels\n"
            f"LED Spacing: {self.led_spacing} mm\n"
            f"Distance: {self.dist} mm\n"
            f"LED List: {self.list_leds}\n"
            f"--------------------"
        )

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
        self.obj = obj
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
            pup_obj = torch.fft.fftshift(torch.fft.fft2(obj_field), dim = (-1, -2))*pupil
            # multiply object's FFT with the pupil stop and take ifft to get measurement
            y[k] = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(pup_obj,  dim=(-1, -2))))
        y = torch.sum(y,0)
        return (y, pup_obj)

        # create list of illumination angles (leds) to turn on one at a time
    
    def forwardFPM(self):
        # multiply by the sample
        obj_field = self.field*self.obj
        # take the fourier transform and center in pupil plane
        pup_obj = torch.fft.fftshift(torch.fft.fft2(obj_field))*self.pupil
        # multiply object's FFT with the pupil stop and take ifft to get measurement
        y = torch.abs(torch.fft.ifft2(torch.fft.fftshift(pup_obj)))
        return (y, pup_obj)
    
    def to(self, device):
        # Move each tensor attribute to the specified device if it exists
        if hasattr(self, 'obj'):
            self.obj = self.obj.to(device)
        if hasattr(self, 'field'):
            self.field = self.field.to(device)
        if hasattr(self, 'pupil'):
            self.pupil = self.pupil.to(device)
        if hasattr(self, 'xygrid'):
            self.xygrid = self.xygrid.to(device)
        if hasattr(self, 'pupilstack'):
            self.pupilstack = self.pupilstack.to(device)
        if hasattr(self, 'illumstack'):
            self.illumstack = self.illumstack.to(device)
        if hasattr(self, 'objstack'):
            self.objstack = self.objstack.to(device)

    def led_ind_to_illum_angle(self, led_ind):
        """
        Convert LED indices to illumination angles.

        Args:
            led_ind (tuple): The LED indices (y, x).

        Returns:
            tuple: The illumination angles (rady, radx).
        """
        led_pos = (led_ind[0] * self.led_spacing, led_ind[1] * self.led_spacing, self.dist)  # units = millimeters, (y, x, z)
        illum_angle = (np.arctan(led_pos[0] / led_pos[2]), np.arctan(led_pos[1] / led_pos[2]))
        return illum_angle

    def createRandomAngleMeasStack(self, list_leds=None, dist=None, led_spacing=None):
        self.list_leds = list_leds
        if self.list_leds is None:  # Handles both empty list and None cases
            raise ValueError("list_leds must be provided")
        # update distance and led spacing if provided
        if dist is not None:
            self.dist = dist
        if led_spacing is not None:
            self.led_spacing = led_spacing

        # create measurement stack
        num_meas = len(self.list_leds)
        measstack = torch.zeros(num_meas, self.Ny, self.Nx)

        for k2 in np.arange(num_meas):
            # take led indices and calculate angle of incidence
            led_ind = self.list_leds[k2]
            illum_angle = self.led_ind_to_illum_angle(led_ind)

            # create illumination field stack for spectrally uniform broadband led
            self.createFixedAngleIllumStack(illum_angle)

            # simulate the forward measurement
            (y, pup_obj) = self.forwardSFPM() 
            
            # plot some example measurements
            if k2 < 5 and plot_flag:
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 4, 1)
                plt.imshow(torch.log10(torch.abs(pup_obj)))
                plt.subplot(1, 4, 2)
                plt.imshow(y, 'gray')
                plt.subplot(1, 4, 3)
                plt.imshow(torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(y)))))
            measstack[k2, :, :] = y
        self.measstack = measstack
        return (measstack, self.list_leds)

class Reconstruction():
    # create a Reconstruction object based on fpm setup
    def __init__(self, fpm_setup, measstack=None, device = 'cpu'):
        self.fpm_setup = fpm_setup
        self.Nw = fpm_setup.Nw
        self.Nx = fpm_setup.Nx
        self.Ny = fpm_setup.Ny
        if measstack is not None: # update measstack if provided
           self.fpm_setup.measstack = measstack
        self.num_meas = len(self.fpm_setup.measstack)
        self.objest = None
        self.device = use_gpu(device)
        self.initRecon()
    
    def __str__(self):
        # Return a string representation of the Reconstruction object
        return (
            "Reconstruction Parameters:\n"
            + "-" * 20 + "\n"
            + f"Number of  measurements: {self.num_meas}\n"
            + f"Device: {self.device}\n"
            + "-" * 20 + "\n"
            + str(self.fpm_setup)
        )

    # initialize the object estimate
    def initRecon(self, obj=None):
        # initialize the object estimate
        if obj is None:
            init_spectrum = torch.ones([self.Nw,1,1]) 
            self.objest = self.fpm_setup.measstack[0,:,:].unsqueeze(0).to(self.device)*init_spectrum.to(self.device)
            self.objest = self.objest/ torch.amax(self.objest) # normalize to max value is 1
        else:
            self.objest = obj

        self.objest.requires_grad = True
        self.losses = []
        return self.objest
    
    def hardthresh(self,x,val):
        return torch.maximum(x,torch.tensor(val))
    
    # set hyperparameters
    def parameters(self, step_size = 1e1, num_iters = 100, loss_type = '2-norm', epochs = 1, opt_type = 'Adam'):
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
        self.epochs = epochs
        self.losses =[]
        self.set_loss(loss_type)
        self.set_optimizer(opt_type)

    def set_loss(self, loss_type):
        if loss_type == 'MSE':
            self.lossfunc = torch.nn.MSELoss()
        if loss_type == '2-norm':
            self.lossfunc = lambda yest, meas: torch.norm(yest-meas)
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
        # move to gpu
        if self.device.type == 'cuda':
            with torch.no_grad():
                self.objest = self.objest.to(self.device)
                self.fpm_setup.measstack = self.fpm_setup.measstack.to(self.device)
                self.fpm_setup.to(self.device)
                
            self.objest.requires_grad = True
        
        # train the objest
        for k3 in np.arange(self.epochs): # loop through epochs
            for k2 in np.arange(self.num_meas): # loop through measurements
                                # print(k1,k2)
                # get relevant actual measurement and move to gpu
                meas = self.fpm_setup.measstack[k2,:,:].double().to(self.device)
                # loop through wavelength 
                # compute illumination angle from led indices
                led_ind = self.fpm_setup.list_leds[k2]   
                illum_angle = self.fpm_setup.led_ind_to_illum_angle(led_ind) 
                # create illumination stack
                self.fpm_setup.createFixedAngleIllumStack(illum_angle)
                self.fpm_setup.illumstack = self.fpm_setup.illumstack.to(self.device)

                for k1 in np.arange(self.num_iters): # loop through iterations

                    # simulate the forward measurement
                    self.fpm_setup.objstack = self.objest


                    (yest, pup_obj) = self.fpm_setup.forwardSFPM()

                    # calculate error, aka loss, and backpropagate
                    error = self.lossfunc(yest,meas)

                    self.losses.append(error.detach().cpu())
                    error.backward()
                    # print(error)

                    # Update the object's reconstruction estimate using the optimizer
                    self.optimizer.step()  # Apply the Adam update to objest
                    self.optimizer.zero_grad()  # Clear gradients after each step

                    if k1 == self.num_iters - 1:
                        try:
                            print(k3, k2, k1)
                            # Visualize the object estimate and its FFT
                            self.visualize(k1,k2,k3)

                        except KeyboardInterrupt:
                            break
        
    def visualize(self, k1, k2, k3):
        # Clear the current figures to update them with new data
        plt.close('all')  # Close all existing figures

        # Create a figure for the loss plot
        loss_fig, ax_loss = plt.subplots(figsize=(8, 6))
        ax_loss.semilogy(self.losses)
        ax_loss.set_title('Loss over time')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')

        # Create a figure with two subplots for the object estimate and its FFT
        fig, (ax_obj, ax_fft) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot the object estimate
        obj2d = np.sum(self.objest.detach().cpu().numpy(), axis=0)  # Sum over wavelengths
        im_obj = ax_obj.imshow(obj2d, cmap='gray')
        ax_obj.set_title('Object 2D Estimate after Epoch {} Meas {} Iter {}'.format(k3+1, k2+1, k1+1))
        fig.colorbar(im_obj, ax=ax_obj, orientation='vertical')  # Add colorbar for objest

        # Plot the FFT of the object estimate
        fftobj2d = np.fft.fftshift(np.fft.fft2(obj2d))
        im_fft = ax_fft.imshow(np.log(np.abs(fftobj2d)), cmap='viridis')  # Plot the magnitude of the FFT
        ax_fft.set_title('FFT of Object 2D Estimate')
        fig.colorbar(im_fft, ax=ax_fft, orientation='vertical')  # Add colorbar for FFT

        # Save the object estimate and FFT figure
        fig.savefig('object_estimate_fft.png', bbox_inches='tight')  # Save with tight layout

        # Display the updated figures
        display.display(loss_fig)
        display.display(fig)
        display.clear_output(wait=True)  # Update display

def debug_plot(obj): #obj is torch tensor
    plt.imshow(np.abs(np.sum(obj.detach().cpu().numpy(), axis=0)))
    plt.colorbar()
    plt.show()

