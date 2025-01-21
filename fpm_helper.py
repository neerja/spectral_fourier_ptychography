import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import image
import math
import os
from IPython import display  # for refreshing plotting
import warnings
import json
from pathlib import Path

import wandb
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Global flag for plotting
plot_flag = True

def use_gpu(gpu=2):
    """
    Set the GPU device for PyTorch and return the device object.

    Args:
        gpu (int): The GPU index to use. Defaults to 2.

    Returns:
        torch.device: The device object for the specified GPU.
    """
    if gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        torch.cuda.set_device(gpu)
        device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')  # Use CPU if no GPU is specified
    torch.cuda.empty_cache()  # Free up space by emptying the cache
    print(device)
    return device

def preprocessObject(im):
    """
    Preprocess the input image to make it square and normalize it.

    Args:
        im (torch.Tensor): The input image tensor.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    # Make image square
    if im.shape[0] < im.shape[1]:
        im = im[:, :im.shape[0]]
    elif im.shape[1] < im.shape[0]:
        im = im[:im.shape[1], :]
        
    # Make dimensions even
    if im.shape[0] % 2 != 0:
        im = im[:-1, :]
    if im.shape[1] % 2 != 0:
        im = im[:, :-1]

    # Normalize to max value of 1
    im = im / torch.amax(im)
    return im

def createlist_led(num_leds=100, minval=-3, maxval=3):
    """
    Create a list of LED positions.  Start random seed at 0 for reproducibility.

    Args:
        num_leds (int): Number of LEDs. Defaults to 100.
        minval (float): Minimum value for LED positions. Defaults to -3.
        maxval (float): Maximum value for LED positions. Defaults to 3.

    Returns:
        np.ndarray: Array of LED positions.
    """
    np.random.seed(0)  # For reproducibility
    list_leds = np.random.uniform(low=minval, high=maxval, size=(num_leds, 2))
    # Append a (0,0) LED for brightfield
    list_leds = np.append(list_leds, np.zeros([1, 2]), axis=0)
    # Sort the list of LEDs by distance from origin
    # important for reconstruction (start with brightest LEDs)
    list_leds = list_leds[np.argsort(np.linalg.norm(list_leds, axis=1))] 
    return list_leds

def create_spiral_leds(num_leds=100, minval=-3, maxval=3, alpha=0.1):
    """
    Create a spiral pattern of LEDs starting from the center (0,0) and spiraling outward
    with consistent spacing between LEDs.

    Args:
        num_leds (int): Number of LEDs. Defaults to 100.
        minval (float): Minimum value for LED positions. Defaults to -3.
        maxval (float): Maximum value for LED positions. Defaults to 3.

    Returns:
        np.ndarray: Array of LED positions in a spiral pattern.
    """
    # Calculate the maximum radius
    max_radius = maxval

    # Create a linear space for the radius
    radius = np.linspace(0, max_radius, num_leds)

    # Use a logarithmic spiral to maintain consistent spacing
    # Adjust this parameter to control the tightness of the spiral
    theta = alpha * np.log(1 + radius)

    # Convert polar coordinates to Cartesian coordinates
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Combine x and y into a single array
    list_leds = np.vstack((x, y)).T

    return list_leds
    

class FPM_setup:
    """
    Class to set up the Fourier Ptychographic Microscopy (FPM) system.  
    It will contain the object, illumination field, pupil stop, and measurement.
    It will also contain the physical parameters associatied with the microsocope (ex: mag, NA, etc).
    """

    def __init__(self, pix_size_camera=None, mag=None, wv=None, na_obj=None, Nx=None, Ny=None, led_spacing=5, dist=75, list_leds=None):
        """
        Initialize the FPM setup with optional parameters.

        Args:
            pix_size_camera (float): Camera pixel size in microns.
            mag (float): Magnification.
            wv (float): Wavelength in microns.
            na_obj (float): Numerical aperture of the objective.
            Nx (int): Number of pixels in x-direction.
            Ny (int): Number of pixels in y-direction.
            led_spacing (float): Spacing between LEDs in mm.
            dist (float): Distance from sample to LED in mm.
            list_leds (list): List of LED positions.
        """
        self.device = torch.device('cpu')
        # Initialize camera pixel size
        if pix_size_camera is None:
            self.pix_size_camera = 4  # Default to 4 microns
        elif isinstance(pix_size_camera, (int, float)):
            self.pix_size_camera = pix_size_camera
        else:
            raise TypeError("pix_size_camera must be a scalar (int or float)")

        # Initialize magnification
        if mag is None:
            self.mag = 1  # Default magnification
        elif isinstance(mag, (int, float)):
            self.mag = mag
        else:
            raise TypeError("mag must be a scalar (int or float)")

        self.pix_size_object = self.pix_size_camera / self.mag  # Calculate object pixel size

        # Initialize wavelength
        if wv is None:
            self.wv = torch.Tensor([500e-3])  # Default to 500 nm
        elif np.isscalar(wv):
            self.wv = torch.Tensor([wv])  # Convert scalar to Tensor
        elif isinstance(wv, np.ndarray):
            self.wv = torch.Tensor(wv)  # Convert NumPy array to Tensor
        elif isinstance(wv, torch.Tensor):
            self.wv = wv
        else:
            raise TypeError("wv must be a scalar, None, or a NumPy array or torch Tensor")

        self.Nw = len(self.wv)  # Number of wavelengths

        # Initialize numerical aperture
        if na_obj is None:
            self.na_obj = 0.05  # Default NA
        elif isinstance(na_obj, (int, float)):
            self.na_obj = na_obj
        else:
            raise TypeError("na_obj must be a scalar (int or float)")

        # Initialize image dimensions
        if Nx is None:
            print('Using stock object data')
            path = '/mnt/neerja-DATA/SpectralFPMData/usafrestarget.jpeg'
            im = torch.from_numpy(image.imread(path))[:, :, 0]  # Use first channel
            im = preprocessObject(im)
            (Ny, Nx) = im.shape
            self.Nx = Nx
            self.Ny = Ny
            self.Npixel = Ny
        else:
            print('Using provided Nx and Ny')
            self.Nx = Nx
            self.Ny = Ny
            self.Npixel = Ny

        # Create xy grid for object space
        self.createXYgrid()

        # Create pupil stack for setup
        self.createPupilStack()

        # Create default uniform spectral object
        self.makeUniformSpectralObject()

        # Initialize LED-related attributes
        self.led_spacing = led_spacing
        self.dist = dist
        self.list_leds = list_leds
        self.list_illums = None # Populated after illumination strategy is set

    def __str__(self):
        """
        Return a string representation of the FPM_setup object.
        """
        base_info = (
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
            f"Total number of angles: {len(self.list_leds) if hasattr(self, 'list_leds') and self.list_leds is not None else 0}\n"
        )
        
        # Add information about illumination configurations
        illum_info = ""
        if hasattr(self, 'list_illums') and self.list_illums is not None:
            illum_info += f"Number of illumination configurations: {len(self.list_illums)}\n"
            illum_info += "Illumination strategy: "
            # Try to identify the illumination strategy
            if all(len(wv_ind) == self.Nw for _, wv_ind in self.list_illums):
                illum_info += "Uniform (all wavelengths per angle)\n"
            elif all(len(wv_ind) == 1 for _, wv_ind in self.list_illums):
                if len(self.list_illums) == len(self.list_leds):
                    illum_info += "Single wavelength per angle (cycling)\n"
                else:
                    illum_info += "Multi wavelength per angle\n"
        elif hasattr(self, 'list_leds') and self.list_leds is not None:
            illum_info += "Using legacy LED list\n"
        
        return base_info + illum_info + "--------------------"

    def makeUniformSpectralObject(self, obj=None):
        """
        Create a uniform spectral object.

        Args:
            obj (torch.Tensor): Optional object data.

        Returns:
            torch.Tensor: The spectral object stack.
        """
        if obj is None:
            path = '/mnt/neerja-DATA/SpectralFPMData/usafrestarget.jpeg'
            im = torch.from_numpy(image.imread(path))[:, :, 0]
        obj = preprocessObject(im)
        spectrum = torch.ones([self.Nw, 1, 1])  # Spectrum vector
        spectral_obj = obj.unsqueeze(0) * spectrum  # Elementwise multiply
        self.objstack = spectral_obj
        self.obj = obj
        return self.objstack

    def createPupilStop(self, wv):
        """
        Create a pupil stop for a given wavelength.

        Args:
            wv (float): Wavelength.

        Returns:
            torch.Tensor: The pupil stop with dtype torch.complex64
        """
        fPupilEdge = self.na_obj / wv  # Edge of pupil
        fMax = 1 / (2 * self.pix_size_object)
        df = 1 / (self.Npixel * self.pix_size_object)
        fvec = np.arange(-fMax, fMax, df)
        fxc, fyc = np.meshgrid(fvec, fvec)
        frc = np.sqrt(fxc**2 + fyc**2)  # Radius of f coordinates
        pupil = np.zeros_like(frc)
        pupil[torch.Tensor(frc) < fPupilEdge] = 1  # Transmit 100% inside edge
        # Convert to complex tensor correctly
        return torch.tensor(pupil, dtype=torch.complex64)

    def createPupilStack(self):
        """
        Create a stack of pupil stops, one for each wavelength channel.

        Returns:
            torch.Tensor: The pupil stack.
        """
        pupilstack = torch.zeros([self.Nw, self.Ny, self.Nx])
        for k in np.arange(self.Nw):
            wv = self.wv[k]
            pupilstack[k, :, :] = self.createPupilStop(wv)
        self.pupilstack = pupilstack
        return self.pupilstack

    def createXYgrid(self):
        """
        Create an xy grid for object space.  Note, depends on camera pixel size and magnification.

        Returns:
            torch.Tensor: The xy grid.
        """
        xvec = np.arange(-self.pix_size_object * self.Nx / 2, self.pix_size_object * self.Nx / 2, self.pix_size_object)
        yvec = np.arange(-self.pix_size_object * self.Ny / 2, self.pix_size_object * self.Ny / 2, self.pix_size_object)
        xx, yy = np.meshgrid(xvec, yvec)
        xygrid = np.array([xx, yy])
        self.xygrid = torch.tensor(xygrid)
        return self.xygrid

    def createIllumField(self, illum_angle, wv):
        """
        Create the illumination field for a given angle and wavelength.

        Args:
            illum_angle (tuple): The illumination angles (rady, radx).
            wv (float): Wavelength.

        Returns:
            torch.Tensor: The illumination field.
        """
        rady = illum_angle[0]
        radx = illum_angle[1]
        k0 = 2 * math.pi / torch.Tensor(wv)
        ky = k0 * math.sin(rady)
        kx = k0 * math.sin(radx)
        field = torch.exp(1j * kx * self.xygrid[1] + 1j * ky * self.xygrid[0]) # not accounting for phase shift due to z
        return field
    
    def createFixedAngleIllumStack(self, illum_angle):
        """
        Deprecated: Use createCustomAngleWavelengthIllumStack instead.
        Create a stack of illumination fields for each wavelength at a fixed angle.

        Args:
            illum_angle (tuple): The illumination angles (rady, radx).

        Returns:
            torch.Tensor: The illumination stack.
        """
        warnings.warn(
            "createFixedAngleIllumStack is deprecated and will be removed in a future version. "
            "Use createCustomAngleWavelengthIllumStack with all wavelength indices instead.",
            DeprecationWarning,
            stacklevel=2
        )
        illumstack = torch.zeros([self.Nw, self.Ny, self.Nx], dtype=torch.complex64)
        for k in np.arange(self.Nw):
            wv = self.wv[k]
            illumstack[k, :, :] = self.createIllumField(illum_angle, wv)
        self.illumstack = illumstack
        return self.illumstack
    
    def createCustomAngleWavelengthIllumStack(self, illum_angle, wv_inds):
        """
        Create an illumination stack for specific wavelength channels at a given angle.
        This allows for selective wavelength illumination patterns.

        Args:
            illum_angle (tuple): The illumination angles (rady, radx).
            wv_inds (tuple): Wavelength indices to illuminate.

        Returns:
            torch.Tensor: The illumination stack with fields only at specified wavelengths.
            The stack has dimensions [Nw, Ny, Nx] where non-specified wavelength 
            channels contain zeros.
        """
        illumstack = torch.zeros([self.Nw, self.Ny, self.Nx], dtype=torch.complex64)
        for k in np.arange(self.Nw):
            if k in wv_inds:
                wv = self.wv[k]
                illumstack[k, :, :] = self.createIllumField(illum_angle, wv)
        self.illumstack = illumstack.to(self.device)
        return self.illumstack

    def forwardSFPM(self):
        """
        Compute the measurement given an object, incident wave field, and pupil stop.

        Returns:
            tuple: The measurement and pupil object.
        """
        y = torch.zeros_like(self.objstack)
        pup_obj = torch.zeros_like(self.objstack, dtype=torch.complex64)
        for k in torch.arange(self.Nw):
            # if illumstack[k] is not all zeros, then use it
            if not torch.all(self.illumstack[k] == 0):
                obj = self.objstack[k]
                pupil = self.pupilstack[k]
                field = self.illumstack[k]
                obj_field = field * obj
                pup_obj[k,:,:] = torch.fft.fftshift(torch.fft.fft2(obj_field), dim=(-1, -2)) * pupil
                y[k,:,:] = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(pup_obj[k,:,:], dim=(-1, -2))))
        y = torch.sum(y, 0) # sum over wavelengths
        pup_obj = torch.sum(pup_obj, 0) # sum over wavelengths
        return (y, pup_obj)

    def computeFourierShift_fromAngle(self, angle, wv, pupil):
        """
        Compute the Fourier shift from an angle.
        """
        rady = angle[0]
        radx = angle[1]
        k0 = 2 * math.pi / torch.Tensor(wv)
        ky = k0 * math.sin(rady)
        kx = k0 * math.sin(radx)
        # shift the pupil by ky, kx
        pupil = torch.roll(pupil, int(ky), dims=0)
        pupil = torch.roll(pupil, int(kx), dims=1)
        return pupil

    def forwardFPM(self):
        """
        Deprecated: Use forwardSFPM instead.
        Compute the measurement for a 2D object and field.

        Returns:
            tuple: The measurement and pupil object.
        """
        warnings.warn(
            "forwardFPM is deprecated and will be removed in a future version. "
            "Use forwardSFPM instead.",
            DeprecationWarning,
            stacklevel=2
        )
        obj_field = self.field * self.obj
        pup_obj = torch.fft.fftshift(torch.fft.fft2(obj_field)) * self.pupil
        y = torch.abs(torch.fft.ifft2(torch.fft.fftshift(pup_obj)))
        return (y, pup_obj)

    def to(self, device):
        """
        Move each tensor attribute to the specified device.

        Args:
            device (torch.device): The device to move tensors to.
        """
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
        self.device = device

    def led_ind_to_illum_angle(self, led_ind):
        """
        Convert LED indices to illumination angles.

        Args:
            led_ind (tuple): The LED indices (y, x).

        Returns:
            tuple: The illumination angles (rady, radx).
        """
        led_pos = (led_ind[0] * self.led_spacing, led_ind[1] * self.led_spacing, self.dist)
        illum_angle = (np.arctan(led_pos[0] / led_pos[2]), np.arctan(led_pos[1] / led_pos[2]))
        return illum_angle
    
    def createSingleWavelengthPerAngleIllumList(self, list_leds=None):
        """
        Create a list of illumination configurations where each angle uses a single wavelength.
        The wavelengths cycle through the available channels using modulo arithmetic.
        
        For example, with 3 wavelengths:
        - First angle uses wavelength 0
        - Second angle uses wavelength 1 
        - Third angle uses wavelength 2
        - Fourth angle uses wavelength 0 again
        And so on...

        Returns:
            list: List of tuples containing (illum_angle, wv_ind) pairs.
            illum_angle is the illumination angle tuple (rady, radx)
            wv_ind is a tuple containing the wavelength indices to use
        """
        if list_leds is not None:
            self.list_leds = list_leds
        self.list_illums = []
        for k in np.arange(len(self.list_leds)):
            illum_angle = self.led_ind_to_illum_angle(self.list_leds[k])
            wv_ind = np.mod(k, self.Nw)  # Cycle through wavelengths
            self.list_illums.append((illum_angle, (wv_ind,)))  # Make wv_ind a tuple
        return self.list_illums
    
    def createMultiWavelengthPerAngleIllumList(self, list_leds=None):
        """
        Create a list of illumination configurations where each angle is used for each wavelength.  
        So that there are Nw*num_leds illumination configurations.

        Returns:
            list: List of tuples containing (illum_angle, wv_ind) pairs.
            illum_angle is the illumination angle tuple (rady, radx)
            wv_ind is a tuple containing the wavelength indices to use
        """
        if list_leds is not None:
            self.list_leds = list_leds
            
        self.list_illums = []
        for k in np.arange(len(self.list_leds)):
            illum_angle = self.led_ind_to_illum_angle(self.list_leds[k])
            for wv_ind in np.arange(self.Nw):
                self.list_illums.append((illum_angle, (wv_ind,)))  # Add comma to make tuple
        return self.list_illums
    
    def createUniformWavelengthPerAngleIllumList(self, list_leds=None):
        """
        Create a list of illumination configurations where each angle uses all wavelengths.

        Returns:
            list: List of tuples containing (illum_angle, wv_ind) pairs.
            illum_angle is the illumination angle tuple (rady, radx)
            wv_ind is a tuple containing the wavelength indices to use
        """

        self.list_illums = []
        if list_leds is not None:
            self.list_leds = list_leds
        for k in np.arange(len(self.list_leds)):
            illum_angle = self.led_ind_to_illum_angle(self.list_leds[k])
            self.list_illums.append((illum_angle, tuple(np.arange(self.Nw))))
        return self.list_illums

    def createMeasStackFromListIllums(self, list_illums = None, visualize=False):
        """
        Create a measurement stack by simulating measurements for each illumination 
        configuration in list_illums.

        Args:
            list_illums (list, optional): List of (illum_angle, wv_ind) pairs. If None, 
                uses self.list_illums.

        Returns:
            torch.Tensor: Measurement stack with dimensions [num_measurements, Ny, Nx].
            Each slice is the simulated measurement for one illumination configuration.
        """
        if list_illums is not None:
            self.list_illums = list_illums
        if self.list_illums is None:
            raise ValueError("No illumination configurations provided. Call create*IllumList first.")
        
        num_meas = len(self.list_illums)
        measstack = torch.zeros(num_meas, self.Ny, self.Nx, device=self.device)
        for k in np.arange(num_meas):
            illum_angle, wv_ind = self.list_illums[k]
            self.createCustomAngleWavelengthIllumStack(illum_angle, wv_ind)
            self.illumstack = self.illumstack.to(self.device)
            (y,pup_obj) = self.forwardSFPM()
            measstack[k, :, :] = y
            if k < 5 and plot_flag:
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 4, 1)
                plt.imshow(torch.log10(torch.abs(pup_obj.detach().cpu())))
                plt.subplot(1, 4, 2)
                plt.imshow(y.cpu(), 'gray')
                plt.subplot(1, 4, 3)
                plt.imshow(torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(y.detach().cpu())))))
                plt.draw()
                plt.pause(0.1)
                plt.show(block=False) # keep plot open in the background
        self.measstack = measstack
        return measstack

    def createRandomAngleMeasStack(self, list_leds=None, dist=None, led_spacing=None):
        """
        Deprecated: Use createUniformWavelengthPerAngleIllumList followed by createMeasStackFromListIllums instead.
        Create a measurement stack with random illumination angles.

        Args:
            list_leds (list): List of LED positions.
            dist (float): Distance from sample to LED.
            led_spacing (float): Spacing between LEDs.

        Returns:
            tuple: The measurement stack and list of LEDs.
        """
        warnings.warn(
            "createRandomAngleMeasStack is deprecated and will be removed in a future version. "
            "Use createUniformWavelengthPerAngleIllumList followed by createMeasStackFromListIllums instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.list_leds = list_leds
        if self.list_leds is None:
            raise ValueError("list_leds must be provided")
        if dist is not None:
            self.dist = dist
        if led_spacing is not None:
            self.led_spacing = led_spacing

        num_meas = len(self.list_leds)
        measstack = torch.zeros(num_meas, self.Ny, self.Nx)

        for k2 in np.arange(num_meas):
            led_ind = self.list_leds[k2]
            illum_angle = self.led_ind_to_illum_angle(led_ind)
            self.createFixedAngleIllumStack(illum_angle)
            (y, pup_obj) = self.forwardSFPM()

            if k2 < 5 or k2 == num_meas-1 and plot_flag:
                plt.figure(figsize=(10, 10))
                #add title onto whole figure
                plt.suptitle(f'Meas Index = {k2}')
                plt.subplot(1, 4, 1)
                plt.title('Pupil')  
                plt.imshow(torch.log10(torch.abs(pup_obj)))
                plt.subplot(1, 4, 2)
                plt.title('Measurement')
                plt.imshow(y, 'gray')
                plt.subplot(1, 4, 3)
                plt.title('Meas FFT')
                plt.imshow(torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(y)))))
            measstack[k2, :, :] = y
        self.measstack = measstack
        return (measstack, self.list_leds)

    def createTile(self, Nw=None):
        """
        Create a tile pattern for the aperture.

        Args:
            Nw (int, optional): Number of wavelength channels. If None, uses self.Nw.

        Returns:
            numpy.ndarray: Tile pattern with values from 1 to Nw.
        """
        if Nw is None:
            Nw = self.Nw
        a = np.arange(1, Nw+1)  # get vector from 1 to Nw
        s = np.ceil(np.sqrt(Nw)).astype(int)  # compute sqrt and round up
        a = np.resize(a, (s, s))  # reshape into square matrix
        return a

    def createAperture(self, tile=None):
        """
        Create an aperture stack where each wavelength has its own region in k-space.
        
        Args:
            tile (numpy.ndarray, optional): Tile pattern to use. If None, creates default tile.

        Returns:
            torch.Tensor: Aperture stack with shape [Nw, Ny, Nx].
            Each slice contains binary mask for that wavelength's allowed k-space region.
        """
        if tile is None:
            tile = self.createTile()
        
        aperture_stack = torch.zeros([self.Nw, self.Ny, self.Nx])
        
        for k in range(self.Nw):
            wv = self.wv[k]
            # Calculate frequency space parameters
            df = 1/(self.Ny * self.pix_size_object)
            f_width = self.na_obj/wv * np.sqrt(2)  # side of largest square that fits in circle pupil
            f_width_ind = int(torch.floor(torch.tensor(f_width/df)))  # Convert to PyTorch operation
            
            # Scale tile to pupil size
            mul_factor = int(f_width_ind/tile.shape[0])  # assume square aperture
            aperture = np.repeat(np.repeat(tile, mul_factor, axis=0), mul_factor, axis=1)
            
            # Pad to full size
            pady = int((self.Ny-f_width_ind)/2) + 1
            padx = int((self.Nx-f_width_ind)/2) + 1
            aperture = np.pad(aperture, ((pady, pady), (padx, padx)))
            aperture = aperture[0:self.Ny, 0:self.Nx]
            
            # Create binary mask for this wavelength
            frame = np.zeros((self.Ny, self.Nx))
            frame[np.where(aperture == k+1)] = 1
            aperture_stack[k, :, :] = torch.tensor(frame)
        self.aperture = aperture
        self.aperture_stack = aperture_stack
        return aperture_stack

    def visualize_pupil_aperture(self, wavelength_index=None):
        """
        Visualize the pupil and aperture configuration.
        
        Args:
            wavelength_index (int, optional): Index of wavelength to visualize. 
                If None, shows sum across all wavelengths.
        """
        if not hasattr(self, 'aperture_stack'):
            raise ValueError("No aperture stack found. Call createAperture first.")
        
        plt.figure(figsize=(15, 4))
        
        if wavelength_index is not None:
            # Show pupil
            plt.subplot(131)
            plt.imshow(self.pupilstack[wavelength_index])
            plt.title(f'Pupil for λ = {int(self.wv[wavelength_index]*1000)}nm')
            plt.colorbar()
            plt.xlabel('kx')
            plt.ylabel('ky')
            
            # Show aperture
            plt.subplot(132)
            plt.imshow(self.aperture_stack[wavelength_index])
            plt.title(f'Aperture for λ = {int(self.wv[wavelength_index]*1000)}nm')
            plt.colorbar()
            plt.xlabel('kx')
            plt.ylabel('ky')
            
            # Show combination
            plt.subplot(133)
            plt.imshow(self.pupilstack[wavelength_index] + self.aperture_stack[wavelength_index])
            plt.title('Pupil + Aperture')
            plt.colorbar()
            plt.xlabel('kx')
            plt.ylabel('ky')
        else:
            # Show sum of pupils
            plt.subplot(131)
            plt.imshow(torch.sum(self.pupilstack, axis=0))
            plt.title('Sum of all pupils')
            plt.colorbar()
            plt.xlabel('kx')
            plt.ylabel('ky')
            
            # Show sum of apertures
            plt.subplot(132)
            plt.imshow(torch.sum(self.aperture_stack, axis=0))
            plt.title('Sum of all apertures')
            plt.colorbar()
            plt.xlabel('kx')
            plt.ylabel('ky')
            
            # Show sum of combination
            plt.subplot(133)
            k=-1
            plt.imshow(self.pupilstack[k,:,:]+self.aperture)
            plt.xlabel('k_x')
            plt.ylabel('k_y')
            plt.title('wv = ' + str(int(self.wv[k]*1000)) + "nm")
        

        
        plt.tight_layout()
        plt.show(block=False)
    
    def updatePupilWithAperture(self):
        """
        Update the pupil stack to include the aperture.

        Returns:
            torch.Tensor: Updated pupil stack.
        """
        self.pupilstack = self.aperture_stack.clone()
        return self.pupilstack
    
    def resetPupil(self):
        """
        Reset the pupil stack to the original pupil stack.
        Uses createPupilStack() to reset.
        
        Returns:
            torch.Tensor: Reset pupil stack.
        """
        self.pupilstack = self.createPupilStack()
        return self.pupilstack

    def visualize_objectfft_coverage(self, list_illums= None):
        """
        Visualize the coverage of the object in k-space.
        """

        if list_illums is None:
            list_illums = self.list_illums
        coverage = torch.zeros(self.Nw, self.Ny, self.Nx, device=self.device)
        df = 1 / (self.Npixel * self.pix_size_object) # assumes square pixels & square object

        # For each illumination angle create pupil circle for each wavelength
        for illum_angle, wv_inds in list_illums:
            for wvind in wv_inds:
                fy = illum_angle[0] /torch.Tensor(self.wv[wvind]) 
                fx = illum_angle[1] /torch.Tensor(self.wv[wvind])
                # convert to indexes
                indx = - int(fx / df) # flip sign so that pupil shifts to left and up for positive angles (DC point moves right and down)
                indy = - int(fy / df)

                # Create circle mask centered at fx, fy
                # get aperture from pupilstack and then shift it to the correct coordinates
                pupil = self.pupilstack[wvind, :, :]
                pupil = torch.roll(pupil, shifts=(int(indy), int(indx)), dims=(0, 1))  # pos angle means shift aperture to left and up
                
                # Set wrapped regions to zero
                if indx > 0:   # if shift to right, then set left side to zero
                    pupil[:, :indx] = 0
                elif indx < 0: # if shift to left, then set right side to zero
                    pupil[:, indx:] = 0
                if indy > 0:   # if shift down, then set top side to zero
                    pupil[:indy, :] = 0
                elif indy < 0: # if shift up, then set bottom side to zero
                    pupil[indy:, :] = 0

                # Add circle to coverage for this wavelength
                coverage[wvind] += pupil
        return coverage


class Reconstruction:
    """
    Class to perform reconstruction based on FPM setup.
    """

    def __init__(self, fpm_setup, measstack=None, device='cpu', recon_cfg=None):
        """
        Initialize the Reconstruction object.

        Args:
            fpm_setup (FPM_setup): The FPM setup object.
            measstack (torch.Tensor): The measurement stack.
            device (str): The device to use ('cpu' or 'cuda').
        """
        self.fpm_setup = fpm_setup
        self.Nw = fpm_setup.Nw
        self.Nx = fpm_setup.Nx
        self.Ny = fpm_setup.Ny
        if measstack is not None:
            self.fpm_setup.measstack = measstack
        self.num_meas = len(self.fpm_setup.measstack)
        self.objest = None
        self.device = device
        self.wandb_run = None  # Initialize wandb run as None
        self.initRecon()
        
        if recon_cfg is not None:
            self.parameters(
                step_size=recon_cfg['step_size'],
                num_iters=recon_cfg['num_iterations'],
                loss_type=recon_cfg['loss_type'],
                epochs=recon_cfg['epochs'],
                opt_type=recon_cfg['optimizer']
            )
            self.wandb_init()

    def __str__(self):
        """
        Return a string representation of the Reconstruction object.
        """
        # Basic reconstruction parameters
        recon_info = (
            "Reconstruction Parameters:\n"
            + "-" * 20 + "\n"
            + f"Number of measurements: {self.num_meas}\n"
            + f"Device: {self.device}\n"
        )
        
        # Optimization parameters if they've been set
        opt_info = ""
        if hasattr(self, 'step_size'):
            opt_info += f"Step size: {self.step_size}\n"
        if hasattr(self, 'num_iters'):
            opt_info += f"Iterations per measurement: {self.num_iters}\n"
        if hasattr(self, 'epochs'):
            opt_info += f"Number of epochs: {self.epochs}\n"
        if hasattr(self, 'optimizer'):
            opt_info += f"Optimizer type: {type(self.optimizer).__name__}\n"
        if hasattr(self, 'lossfunc'):
            if isinstance(self.lossfunc, torch.nn.Module):
                loss_name = type(self.lossfunc).__name__
            else:
                loss_name = "2-norm"
            opt_info += f"Loss function: {loss_name}\n"
        
        # Add FPM setup information
        setup_info = "-" * 20 + "\n" + str(self.fpm_setup)
        
        return recon_info + opt_info + setup_info

    def initRecon(self, obj=None):
        """
        Initialize the object estimate.

        Args:
            obj (torch.Tensor): Optional initial object estimate.

        Returns:
            torch.Tensor: The initialized object estimate.
        """
        if obj is None:
            init_spectrum = torch.ones([self.Nw, 1, 1])
            self.objest = self.fpm_setup.measstack[0, :, :].unsqueeze(0).to(self.device) * init_spectrum.to(self.device)
            self.objest = self.objest / torch.amax(self.objest)  # Normalize to max value of 1
        else:
            self.objest = obj

        self.objest.requires_grad = True
        self.losses = []
        return self.objest

    def hardthresh(self, x, val):
        """
        Apply hard thresholding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            val (float): The threshold value.

        Returns:
            torch.Tensor: The thresholded tensor.
        """
        return torch.maximum(x, torch.tensor(val))

    def parameters(self, step_size=1e1, num_iters=100, loss_type='2-norm', epochs=1, opt_type='Adam'):
        """
        Set hyperparameters for the reconstruction.

        Args:
            step_size (float): The step size for optimization.
            num_iters (int): The number of iterations per measurement.
            loss_type (str): The type of loss function ('MSE' or '2-norm').
            epochs (int): The number of epochs.
            opt_type (str): The type of optimizer ('Adam').
        """
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
        self.losses = []
        self.set_loss(loss_type)
        self.set_optimizer(opt_type)

    def set_loss(self, loss_type):
        """
        Set the loss function for the reconstruction.

        Args:
            loss_type (str): The type of loss function ('MSE' or '2-norm').

        Returns:
            callable: The loss function.
        """
        if loss_type == 'MSE':
            self.lossfunc = lambda yest, meas, **kwargs: torch.nn.MSELoss(yest - meas)
        elif loss_type == '2-norm':
            self.lossfunc = lambda yest, meas, **kwargs: torch.norm(yest - meas)
        else:
            raise ValueError("Loss type not recognized")
        return self.lossfunc


    def set_optimizer(self, opt_type):
        """
        Set the optimizer for the reconstruction.

        Args:
            opt_type (str): The type of optimizer ('Adam').

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        if opt_type == 'Adam':
            self.optimizer = torch.optim.Adam([self.objest], lr=self.step_size)
        else:
            raise ValueError("Optimizer type not recognized")
        return self.optimizer

    def train(self, visualize=True):
        """
        Train the object estimate using the specified parameters.
        """
        try:
            if self.device.type == 'cuda':
                with torch.no_grad():
                    self.objest = self.objest.to(self.device)
                    self.fpm_setup.measstack = self.fpm_setup.measstack.to(self.device)
                    self.fpm_setup.to(self.device)

            self.objest.requires_grad = True
            
            # check that self.num_meas and len(self.fpm_setup.list_illums) are the same if list_illums is not None  
            if self.fpm_setup.list_illums is not None:
                if self.num_meas != len(self.fpm_setup.list_illums):
                    raise ValueError("Number of measurements and list_illums must be the same")
            
            for k3 in np.arange(self.epochs):
                for k2 in np.arange(self.num_meas):
                    meas = self.fpm_setup.measstack[k2, :, :].double().to(self.device)
                    try:
                        illum_angle, wv_ind = self.fpm_setup.list_illums[k2]
                        self.fpm_setup.createCustomAngleWavelengthIllumStack(illum_angle, wv_ind)
                    except: # backward compatibility for no list_illums
                        led_ind = self.fpm_setup.list_leds[k2]
                        illum_angle = self.fpm_setup.led_ind_to_illum_angle(led_ind)
                        self.fpm_setup.createFixedAngleIllumStack(illum_angle)

                    self.fpm_setup.illumstack = self.fpm_setup.illumstack.to(self.device)

                    for k1 in np.arange(self.num_iters):
                        self.fpm_setup.objstack = self.objest
                        (yest, pup_obj) = self.fpm_setup.forwardSFPM()
                        error = self.lossfunc(yest, meas)
                        self.losses.append(error.detach().cpu())
                        error.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if self.wandb_active:
                            self.wandb_logloss()

                        if k1 == self.num_iters - 1:
                            try:
                                print(k3, k2, k1)
                                if visualize:
                                    self.visualize(k1, k2, k3)
                                if self.wandb_active:
                                    self.wandb_logplots()
                                    metrics_log = self.compute_metrics()
                                    for wavelength, metrics in metrics_log.items():
                                        # Log all metrics for this wavelength in a single step
                                        self.wandb_run.log({f"{wavelength}/{metric_name}": value 
                                                          for metric_name, value in metrics.items()})
                            except KeyboardInterrupt:
                                break

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            if self.wandb_active:
                self.wandb_finish()
            raise  # Re-raise the KeyboardInterrupt after cleanup


    def visualize(self, k1, k2, k3):
        """
        Visualize the object estimate and its FFT.

        Args:
            k1 (int): Current iteration.
            k2 (int): Current measurement index.
            k3 (int): Current epoch.
        """
        plt.close('all')

        # Loss plot
        loss_fig = plt.figure(figsize=(8, 6))
        ax_loss = loss_fig.add_subplot(111)
        ax_loss.semilogy(self.losses)
        ax_loss.set_title('Loss over time')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        
        # Draw and show loss plot
        loss_fig.tight_layout()
        plt.draw()
        plt.pause(0.1)
        plt.show(block=False)

        # Object and FFT plot
        fig = plt.figure(figsize=(16, 6))
        
        # Object subplot
        ax_obj = fig.add_subplot(121)
        obj2d = np.sum(self.objest.detach().cpu().numpy(), axis=0)
        im_obj = ax_obj.imshow(obj2d, cmap='gray')
        ax_obj.set_title(f'Object 2D Estimate after Epoch {k3+1} Meas {k2+1} Iter {k1+1}')
        plt.colorbar(im_obj, ax=ax_obj, orientation='vertical')

        # FFT subplot
        ax_fft = fig.add_subplot(122)
        fftobj2d = np.fft.fftshift(np.fft.fft2(obj2d))
        im_fft = ax_fft.imshow(np.log(np.abs(fftobj2d)), cmap='viridis')
        ax_fft.set_title('FFT of Object 2D Estimate')
        plt.colorbar(im_fft, ax=ax_fft, orientation='vertical')

        # Save figure
        fig.savefig('object_estimate_fft.png', bbox_inches='tight')
        
        # Draw and show object/FFT plot
        fig.tight_layout()
        plt.draw()
        plt.pause(0.1)
        plt.show(block=False)
        
        # Clean up
        plt.close(loss_fig)
        plt.close(fig)

    def wandb_init(self):
        """
        Initialize wandb logging and log important reconstruction parameters.
        """
        self.wandb_active = True
        self.wandb_run = wandb.init(project="Spectral_FPM", config={
            "learning_rate": self.step_size,
            "num_iters": self.num_iters,
            "epochs": self.epochs,
            "opt_type": type(self.optimizer).__name__,
            "loss_function": type(self.lossfunc).__name__ if isinstance(self.lossfunc, torch.nn.Module) else "2-norm",
            "device": str(self.device),
            "num_measurements": self.num_meas,
            "fpm_setup_info": str(self.fpm_setup),
        })
        self.wandb_run_id = self.wandb_run.id

        # log the led plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(self.fpm_setup.list_leds[:, 1], self.fpm_setup.list_leds[:, 0])
        ax.set_aspect('equal', 'box')
        ax.set_title('LED Locations')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        self.wandb_run.log({"LED Plot": wandb.Image(fig)})
        plt.close(fig)

        # log the coverage plot
        coverage = self.fpm_setup.visualize_objectfft_coverage(self.fpm_setup.list_illums)
        fig, axes = plt.subplots(1, self.fpm_setup.Nw, figsize=(4 * self.fpm_setup.Nw, 4))
        for wvind in range(self.fpm_setup.Nw):
            if self.fpm_setup.Nw == 1:
                ax = axes
            else:
                ax = axes[wvind]
            im = ax.imshow(coverage[wvind].cpu())
            ax.set_title(f'Fourier Coverage for wavelength {self.fpm_setup.wv[wvind] * 1000:.0f} nm')
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

        plt.tight_layout()
        self.wandb_run.log({"Fourier Coverage Plot": wandb.Image(fig)})
        plt.close(fig)

    def wandb_logloss(self):
        """
        Log the loss to wandb.
        """
        if self.wandb_run is not None:
            self.wandb_run.log({"Loss": self.losses[-1]})

    def wandb_logplots(self):
        """
        Log the object estimate, its zoomed central region, and its FFT to wandb.
        """
        if self.wandb_run is not None:
            obj2d = np.sum(self.objest.detach().cpu().numpy(), axis=0)
            fig = self.plot_object_estimate(obj2d, show_plot=False)  # Don't show the plot
            self.wandb_run.log({"XY Object Estimate": wandb.Image(fig)})
            plt.close(fig)

    def wandb_finish(self):
        """
        Finish wandb logging.
        """
        if self.wandb_run is not None:
            for k in range(self.Nw):
                fig = self.plot_object_estimate(self.objest.detach().cpu().numpy()[k,:,:])
                self.wandb_run.log({"Final Recon for Wavelength {}".format(k): wandb.Image(fig)})
                plt.close(fig)
            self.wandb_run.finish()
            self.wandb_run = None  # Clear the run after finishing
        self.wandb_active = False

    def plot_object_estimate(self, obj2d, show_plot=False):
        """
        Plot the object estimate, its zoomed central region, and its FFT.

        Args:
            obj2d (torch.Tensor): The 2D object estimate.
            show_plot (bool): Whether to display the plot window.
        """
        # Create figure without displaying
        plt.ioff()  # Turn off interactive mode
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Full image
        im1 = ax1.imshow(obj2d, cmap='gray')
        ax1.set_title('Full Recon Image')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax1)
        
        # Zoomed central region
        center_y, center_x = obj2d.shape[0] // 2, obj2d.shape[1] // 2
        zoom_size = 200  # Adjust this value to change zoom level
        zoom_region = obj2d[center_y - zoom_size // 2:center_y + zoom_size // 2,
                            center_x - zoom_size // 2:center_x + zoom_size // 2]
        im2 = ax2.imshow(zoom_region, cmap='gray')
        ax2.set_title('Zoomed Central Region')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax2)

        # FFT of the object estimate
        fftobj2d = np.fft.fftshift(np.fft.fft2(obj2d))
        im3 = ax3.imshow(np.log(np.abs(fftobj2d)), cmap='viridis')
        ax3.set_title('FFT of Object 2D Estimate')
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax3)

        plt.tight_layout()
        
        if show_plot:
            plt.ion()  # Turn interactive mode back on
            plt.show(block=False)

        return fig
        

    def compute_metrics(self):
        """
        Compute the SSIM, PSNR, and MSE between the reconstruction and the ground truth.
        
        Returns:
            dict: Dictionary containing the computed metrics for each wavelength:
                - 'ssim': Structural Similarity Index (higher is better, max 1)
                - 'psnr': Peak Signal-to-Noise Ratio in dB (higher is better)
                - 'mse': Mean Squared Error (lower is better)
                - 'mae': Mean Absolute Error (lower is better)
                
        Notes:
            - Requires ground truth object to be stored in self.fpm_setup.obj
            - Metrics are computed separately for each wavelength
            - All metrics are computed on normalized images (0 to 1 range)
        """
        import torch.nn.functional as F
        from torchmetrics.functional import structural_similarity_index_measure as ssim
        from torchmetrics.functional import peak_signal_noise_ratio as psnr
        
        # Check if ground truth exists
        if not hasattr(self.fpm_setup, 'obj'):
            raise ValueError("Ground truth object not found in FPM setup")
        
        # Get reconstruction and ground truth
        recon_est = self.objest.detach()
        ground_truth = self.fpm_setup.obj.to(recon_est.device)
        
        # Normalize ground truth
        ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())
        
        metrics_log = {}
        for k in range(self.Nw):
            # Normalize reconstruction for current wavelength
            curr_recon = recon_est[k,:,:]
            curr_recon = (curr_recon - curr_recon.min()) / (curr_recon.max() - curr_recon.min())
            
            # Add batch and channel dimensions required by metrics
            curr_recon = curr_recon.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
            curr_ground_truth = ground_truth.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
            
            # Compute metrics
            metrics_log[f'wavelength_{self.fpm_setup.wv[k] * 1000:.0f}nm'] = {
                'ssim': float(ssim(curr_recon, curr_ground_truth, data_range=1.0)),
                'psnr': float(psnr(curr_recon, curr_ground_truth, data_range=1.0)),
                'mse': float(F.mse_loss(curr_recon, curr_ground_truth)),
                'mae': float(F.l1_loss(curr_recon, curr_ground_truth))
            }
            # compute total metrics
            curr_recon = recon_est.sum(dim=0).detach()  # Sum across wavelengths
            # Normalize reconstruction
            curr_recon = (curr_recon - curr_recon.min()) / (curr_recon.max() - curr_recon.min())
            
            # Add batch and channel dimensions required by metrics
            curr_recon = curr_recon.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
            curr_ground_truth = ground_truth.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
            
            metrics_log['summed_wavelengths'] = {
                'ssim': float(ssim(curr_recon, curr_ground_truth, data_range=1.0)),
                'psnr': float(psnr(curr_recon, curr_ground_truth, data_range=1.0)),
                'mse': float(F.mse_loss(curr_recon, curr_ground_truth)),
                'mae': float(F.l1_loss(curr_recon, curr_ground_truth))
            }
        
        # # Log final metrics to wandb if active
        # if self.wandb_active:
        #     # Log final metrics directly to summary
        #     for wavelength, metrics in metrics_log.items():
        #         for metric_name, value in metrics.items():
        #             self.wandb_run.summary[f"{wavelength}/{metric_name}"] = value

        return metrics_log

def debug_plot(obj): 
    """
    Plot the absolute sum of the object tensor.

    Args:
        obj (torch.Tensor): The object tensor.
    """
    if obj.ndim == 3:
        plt.imshow(np.abs(np.sum(obj.detach().cpu().numpy(), axis=0)))
    else:
        plt.imshow(np.abs(obj.detach().cpu().numpy()))
    plt.colorbar()
    plt.show()


class SparseReconstruction(Reconstruction):
    """
    Class to perform sparse reconstruction based on FPM setup by incorporating L1 loss on the object estimate.
    """
    def set_regularizer(self, reg_type='none'):
        """
        Set the regularizer for the reconstruction.
        """
        if reg_type == 'none':
            self.regularizer = None
        elif reg_type == 'L1':
            self.regularizer = lambda objest: torch.norm(objest, p=1)
        elif reg_type == 'L2':
            self.regularizer = lambda objest: torch.norm(objest, p=2)
        else:
            raise ValueError("Regularizer type not recognized")
        return self.regularizer
    
    def parameters(self, step_size=1e1, num_iters=100, loss_type='2-norm', epochs=1, opt_type='Adam', reg_type='L1', tau_reg = 1e-3):
        """
        Initialize the parameters for the reconstruction.
        """
        self.set_regularizer(reg_type) # need to set the regularizer first before parameters sets lossfunc
        super().parameters(step_size, num_iters, loss_type, epochs, opt_type)
        self.tau_reg = tau_reg
    
    def set_loss(self, loss_type):
        """
        Set the loss function for the reconstruction.
        """
        super().set_loss(loss_type)
        if self.regularizer is not None:
        # Define a new loss function that includes the regularizer
            data_lossfunc = self.lossfunc
            self.lossfunc = lambda yest, meas, objest, **kwargs: data_lossfunc(yest, meas) + self.tau_reg * self.regularizer(objest)
        return self.lossfunc

    def wandb_init(self):
        """
        Initialize wandb logging and log important reconstruction parameters.
        """
        super().wandb_init()
        wandb.log({"tau_reg": self.tau_reg})


def save_simulation_results(fpm_setup, recon, save_path):
    """
    Save the simulation results to disk.
    
    Args:
        fpm_setup (FPM_setup): The FPM setup object
        recon (Reconstruction): The reconstruction object
        save_dir (str): Directory to save results
        run_name (str): Name of the run
    """ 
    # Save reconstructed object
    obj2d = np.sum(recon.objest.detach().cpu().numpy(), axis=0)
    np.save(save_path / 'reconstructed_object.npy', recon.objest.detach().cpu().numpy())
    
    # Save loss history
    np.save(save_path / 'loss_history.npy', np.array(recon.losses))
    
    # Save final visualization
    # Create and save visualization using plot_object_estimate
    fig = recon.plot_object_estimate(obj2d)
    fig.savefig(save_path / 'final_reconstruction_sum.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    # save image for each wavelength
    for k in range(fpm_setup.Nw):
        fig = recon.plot_object_estimate(recon.objest.detach().cpu().numpy()[k,:,:])
        fig.savefig(save_path / f'final_reconstruction_wavelength_{k}.png', bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    plt.savefig(save_path / 'final_reconstruction.png')
    plt.close()

    if recon.wandb_active:
        recon.wandb_finish()
