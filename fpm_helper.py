
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import image
import math
import os
from IPython import display # for refreshing plotting


def use_gpu(gpu=2):
    if gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        torch.cuda.set_device(gpu)
        device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu') # uncomment this line to use cpu
    torch.cuda.empty_cache()  #empty any variables in cache to free up space
    # print(device)
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
        else:
            self.pix_size_camera = pix_size_camera
        
        # Handle magnification, default to 1 if not provided
        if mag is None:
            mag = 1  # Default magnification if no argument is passed
        
        self.pix_size_object = self.pix_size_camera / mag  # Calculate based on magnification
        
        # Handle wavelength, default to 500e-3 if not provided
        if wv is None:
            self.wv = np.array(500e-3)  # micron
            self.wv = np.expand_dims(self.wv, axis=0) 
            self.Nw = 1  # Number of wavelengths
        else:
            self.wv = wv
            self.Nw = len(self.wv) # Number of wavelengths
        
        # Handle NA, default to 0.05 if not provided
        if na_obj is None:
            self.na_obj = 0.05  # Default NA
        else:
            self.na_obj = na_obj
        if Nx is None:  # use stock object data if no Nx Ny info is provided
            path = '/mnt/neerja-DATA/SpectralFPMData/usafrestarget.jpeg'
            im = torch.from_numpy(image.imread(path))[:,:,0] # pick out the first channel
            im = preprocessObject(im)
            (Ny,Nx) = im.shape
            self.Nx = Nx
            self.Ny = Ny
            self.Npixel = Ny
        else:
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
        pupil[frc<fPupilEdge] = 1 # make everything inside fPupilEdge transmit 100%
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
        k0 = 2*math.pi/wv
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
        for k in torch.arange(self.Nw):
            obj = self.objstack[k]
            pupil = self.pupilstack[k]
            field = self.illumstack[k]
            obj_field = field*obj
            # take the fourier transform and center in pupil plane
            pup_obj = torch.fft.fftshift(torch.fft.fft2(obj_field))*pupil
            # multiply object's FFT with the pupil stop and take ifft to get measurement
            if k==0:
                y = torch.abs(torch.fft.ifft2(torch.fft.fftshift(pup_obj)))
            else:
                y = y+torch.abs(torch.fft.ifft2(torch.fft.fftshift(pup_obj)))
            # subsample according to pixel size on camera?
        return (y, pup_obj)

        # create list of illumination angles (leds) to turn on one at a time


def createRandomAngleMeasStack(fpm_setup):
    d = 75 # distance away led to object
    led_spacing = 5 # roughly 5 mm apart LEDs
    list_leds = createlist_led(100,-3,3) 
    # create measurement stack
    num_meas = len(list_leds) 
    measstack = torch.zeros(num_meas,fpm_setup.Ny,fpm_setup.Nx)

    for k2 in np.arange(len(list_leds)):
        # take led indices and calculate angle of incidence
        led_ind = list_leds[k2]  
        led_pos = (led_ind[0]*led_spacing,led_ind[1]*led_spacing,d) # units = millimeters, (y,x,z)
        illum_angle = (np.arctan(led_pos[0]/led_pos[2]), np.arctan(led_pos[1]/led_pos[2])) 

        # create illumination field stack 
        illumstack = torch.zeros([fpm_setup.Nw,fpm_setup.Ny,fpm_setup.Nx],dtype = torch.complex64)  # use complex dtype
        for k1 in np.arange(fpm_setup.Nw):  # iterate over wavelengths
            wv = fpm_setup.wv[k1]
            illumstack[k1,:,:] = fpm_setup.createIllumField(illum_angle,wv)
        fpm_setup.illumstack = illumstack
        # simulate the forward measurement
        (y, pup_obj) = fpm_setup.forwardSFPM()
        measstack[k2,:,:] = y
    return (measstack, list_leds)