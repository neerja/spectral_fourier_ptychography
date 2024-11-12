import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import image
import math
import os
from IPython import display  # for refreshing plotting

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
    (Ny, Nx) = im.shape
    # Make image square
    if Ny < Nx:
        im = im[:, :Ny]
    elif Nx < Ny:
        im = im[:Nx, :]
    # Make dimensions even
    if Ny % 2 != 0:
        im = im[:-1, :]
    if Nx % 2 != 0:
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
    list_leds = list_leds[np.argsort(np.linalg.norm(list_leds, axis=1))]
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
            torch.Tensor: The pupil stop.
        """
        fPupilEdge = self.na_obj / wv  # Edge of pupil
        fMax = 1 / (2 * self.pix_size_object)
        df = 1 / (self.Npixel * self.pix_size_object)
        fvec = np.arange(-fMax, fMax, df)
        fxc, fyc = np.meshgrid(fvec, fvec)
        frc = np.sqrt(fxc**2 + fyc**2)  # Radius of f coordinates
        pupil = np.zeros_like(frc)
        pupil[torch.Tensor(frc) < fPupilEdge] = 1  # Transmit 100% inside edge
        return torch.Tensor(pupil)

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
        field = torch.exp(1j * kx * self.xygrid[1] + 1j * ky * self.xygrid[0])
        return field
    
    def createFixedAngleIllumStack(self, illum_angle):
        """
        Create a stack of illumination fields for each wavelength at a fixed angle.

        Args:
            illum_angle (tuple): The illumination angles (rady, radx).

        Returns:
            torch.Tensor: The illumination stack.
        """
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
        self.illumstack = illumstack
        return self.illumstack

    def forwardSFPM(self):
        """
        Compute the measurement given an object, incident wave field, and pupil stop.

        Returns:
            tuple: The measurement and pupil object.
        """
        y = torch.zeros_like(self.objstack)
        for k in torch.arange(self.Nw):
            obj = self.objstack[k]
            pupil = self.pupilstack[k]
            field = self.illumstack[k]
            obj_field = field * obj
            pup_obj = torch.fft.fftshift(torch.fft.fft2(obj_field), dim=(-1, -2)) * pupil
            y[k] = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(pup_obj, dim=(-1, -2))))
        y = torch.sum(y, 0)
        return (y, pup_obj)

    def forwardFPM(self):
        """
        Compute the measurement for a single wavelength.

        Returns:
            tuple: The measurement and pupil object.
        """
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
    
    def createSingleWavelengthPerAngleIllumList(self):
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

        self.list_illums = []
        for k in np.arange(len(self.list_leds)):
            illum_angle = self.led_ind_to_illum_angle(self.list_leds[k])
            wv_ind = np.mod(k, self.Nw)  # Cycle through wavelengths
            self.list_illums.append((illum_angle, (wv_ind)))
        return self.list_illums
    
    def createMultiWavelengthPerAngleIllumList(self):
        """
        Create a list of illumination configurations where each angle is used for each wavelength.  
        So that there are Nw*num_leds illumination configurations.

        Returns:
            list: List of tuples containing (illum_angle, wv_ind) pairs.
            illum_angle is the illumination angle tuple (rady, radx)
            wv_ind is a tuple containing the wavelength indices to use
        """
        self.list_illums = []
        for k in np.arange(len(self.list_leds)):
            illum_angle = self.led_ind_to_illum_angle(self.list_leds[k])
            for wv_ind in np.arange(self.Nw):
                self.list_illums.append((illum_angle, (wv_ind)))
        return self.list_illums
    
    def createUniformWavelengthPerAngleIllumList(self):
        """
        Create a list of illumination configurations where each angle uses all wavelengths.

        Returns:
            list: List of tuples containing (illum_angle, wv_ind) pairs.
            illum_angle is the illumination angle tuple (rady, radx)
            wv_ind is a tuple containing the wavelength indices to use
        """

        self.list_illums = []
        for k in np.arange(len(self.list_leds)):
            illum_angle = self.led_ind_to_illum_angle(self.list_leds[k])
            self.list_illums.append((illum_angle, tuple(np.arange(self.Nw))))
        return self.list_illums

    def createMeasStackFromListIllums(self):
        """
        Create a measurement stack by simulating measurements for each illumination 
        configuration in list_illums.

        Each configuration specifies which angle and wavelength(s) to use.
        The forward model is applied to generate the expected measurement for
        each configuration.

        Returns:
            torch.Tensor: Measurement stack with dimensions [num_measurements, Ny, Nx].
            Each slice is the simulated measurement for one illumination configuration.
        """
        num_meas = len(self.list_illums)
        measstack = torch.zeros(num_meas, self.Ny, self.Nx)
        for k in np.arange(num_meas):
            illum_angle, wv_ind = self.list_illums[k]
            self.createCustomAngleWavelengthIllumStack(illum_angle, wv_ind)
            (y, _) = self.forwardSFPM()
            measstack[k, :, :] = y
        self.measstack = measstack
        return measstack

    def createRandomAngleMeasStack(self, list_leds=None, dist=None, led_spacing=None):
        """
        Create a measurement stack with random illumination angles.

        Args:
            list_leds (list): List of LED positions.
            dist (float): Distance from sample to LED.
            led_spacing (float): Spacing between LEDs.

        Returns:
            tuple: The measurement stack and list of LEDs.
        """
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
    
    def createIllumStack(self, illum_angle, channels):
        """
        Create an illumstack for a given angle and wavelength channels.
        Start with an illumstack of zeros, then fill in the appropriate illumination field for each wavelength.
        Channels is a list of wavelength indices to use. 
        """

class Reconstruction:
    """
    Class to perform reconstruction based on FPM setup.
    """

    def __init__(self, fpm_setup, measstack=None, device='cpu'):
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
        self.device = use_gpu(device)
        self.initRecon()

    def __str__(self):
        """
        Return a string representation of the Reconstruction object.
        """
        return (
            "Reconstruction Parameters:\n"
            + "-" * 20 + "\n"
            + f"Number of measurements: {self.num_meas}\n"
            + f"Device: {self.device}\n"
            + "-" * 20 + "\n"
            + str(self.fpm_setup)
        )

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
            self.lossfunc = torch.nn.MSELoss()
        elif loss_type == '2-norm':
            self.lossfunc = lambda yest, meas: torch.norm(yest - meas)
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

    def train(self):
        """
        Train the object estimate using the specified parameters.
        """
        if self.device.type == 'cuda':
            with torch.no_grad():
                self.objest = self.objest.to(self.device)
                self.fpm_setup.measstack = self.fpm_setup.measstack.to(self.device)
                self.fpm_setup.to(self.device)

            self.objest.requires_grad = True

        for k3 in np.arange(self.epochs):
            for k2 in np.arange(self.num_meas):
                meas = self.fpm_setup.measstack[k2, :, :].double().to(self.device)
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

                    if k1 == self.num_iters - 1:
                        try:
                            print(k3, k2, k1)
                            self.visualize(k1, k2, k3)
                        except KeyboardInterrupt:
                            break

    def visualize(self, k1, k2, k3):
        """
        Visualize the object estimate and its FFT.

        Args:
            k1 (int): Current iteration.
            k2 (int): Current measurement index.
            k3 (int): Current epoch.
        """
        plt.close('all')

        loss_fig, ax_loss = plt.subplots(figsize=(8, 6))
        ax_loss.semilogy(self.losses)
        ax_loss.set_title('Loss over time')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')

        fig, (ax_obj, ax_fft) = plt.subplots(1, 2, figsize=(16, 6))
        obj2d = np.sum(self.objest.detach().cpu().numpy(), axis=0)
        im_obj = ax_obj.imshow(obj2d, cmap='gray')
        ax_obj.set_title('Object 2D Estimate after Epoch {} Meas {} Iter {}'.format(k3+1, k2+1, k1+1))
        fig.colorbar(im_obj, ax=ax_obj, orientation='vertical')

        fftobj2d = np.fft.fftshift(np.fft.fft2(obj2d))
        im_fft = ax_fft.imshow(np.log(np.abs(fftobj2d)), cmap='viridis')
        ax_fft.set_title('FFT of Object 2D Estimate')
        fig.colorbar(im_fft, ax=ax_fft, orientation='vertical')

        fig.savefig('object_estimate_fft.png', bbox_inches='tight')

        display.display(loss_fig)
        display.display(fig)
        display.clear_output(wait=True)

def debug_plot(obj): 
    """
    Plot the absolute sum of the object tensor.

    Args:
        obj (torch.Tensor): The object tensor.
    """
    plt.imshow(np.abs(np.sum(obj.detach().cpu().numpy(), axis=0)))
    plt.colorbar()
    plt.show()

