#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import wandb
from fpm_helper import FPM_setup, Reconstruction, createlist_led, create_spiral_leds, use_gpu, save_simulation_results
import matplotlib.pyplot as plt

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Spectral FPM simulation with config file')
    parser.add_argument('config_path', type=str, help='Path to the configuration file (JSONC)')

    return parser.parse_args()

def run_spectral_fpm_simulation(config_path):
    """
    Run a Spectral Fourier Ptychographic Microscopy simulation using parameters from a config file.
    
    Args:
        config_path (str): Path to the configuration file in JSONC format
        output_dir (str): Directory to save results
        use_wandb (bool): Whether to use Weights & Biases logging
        
    Returns:
        tuple: (FPM_setup object, Reconstruction object) containing the setup and reconstruction results
    """
    # Load and parse config file

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    print("Setting up FPM simulation...")
    
    # Extract microscope parameters
    micro_cfg = config['microscope']
    mag = micro_cfg['magnification']
    pix_size_camera = micro_cfg['pixel_size_camera']
    na_obj = micro_cfg['na_objective']
    
    # Create wavelength range
    wv_cfg = micro_cfg['wavelength']
    wv_range = np.arange(wv_cfg['start'], wv_cfg['end'], wv_cfg['step'])
    print(f"Using wavelength range: {wv_range} microns")
    
    # Create FPM setup
    fpm_setup = FPM_setup(
        pix_size_camera=pix_size_camera,
        mag=mag,
        wv=wv_range,
        na_obj=na_obj,
        led_spacing=config['led_array']['spacing'],
        dist=config['led_array']['distance']
    )
    
    # Create LED array
    print(f"Creating {config['led_array']['pattern']} LED pattern...")
    if config['led_array']['pattern'] == 'random':
        list_leds = createlist_led(
            num_leds=config['led_array']['num_leds'],
            minval=config['led_array']['min_val'],
            maxval=config['led_array']['max_val']
        )
    elif config['led_array']['pattern'] == 'spiral':
        list_leds = create_spiral_leds(
            num_leds=config['led_array']['num_leds'],
            minval=config['led_array']['min_val'],
            maxval=config['led_array']['max_val'],
            alpha=config['led_array']['alpha']
        )
    else:
        raise ValueError(f"Unsupported LED pattern: {config['led_array']['pattern']}")
    
    fpm_setup.list_leds = list_leds
    
    
    # Create illumination list and measurement stack
    print("Creating illumination configurations...")
    fpm_setup.createUniformWavelengthPerAngleIllumList()
    fpm_setup.createMeasStackFromListIllums()
    
    # Initialize reconstruction
    print("Initializing reconstruction...")
    recon_cfg = config['reconstruction']
    device = torch.device(recon_cfg['device'])
    if recon_cfg['device'] == 'cuda':
        device = use_gpu(recon_cfg['gpu_index'])
    
    recon = Reconstruction(
        fpm_setup=fpm_setup,
        device=device, recon_cfg=recon_cfg
        )
    
    run_id = recon.wandb_run_id
    output_path = Path(config['logging']['save_dir'], config['logging']['run_name'], run_id)
    output_path.mkdir(parents=True, exist_ok=True)
    # save config file
    with open(output_path / f'config_{run_id}.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Run reconstruction
    print("Starting reconstruction...")
    try:
        recon.train(visualize=config['logging']['visualize'])
    except KeyboardInterrupt:
        print("\nReconstruction interrupted by user. Saving current results...")
        save_simulation_results(fpm_setup, recon, output_path)
        sys.exit(0)
    except Exception as e:
        print(f"Error during reconstruction: {e}.  Saving current results...")
        save_simulation_results(fpm_setup, recon, output_path)
        raise  # Re-raise the exception after saving results
    
    # Save results
    print("Saving results...")
    save_simulation_results(fpm_setup, recon, output_path)
    
    print(f"Simulation complete! Results saved to: {output_path}")
    return fpm_setup, recon

def show_debug_plot(data, title=""):
    """
    Display a plot with explicit figure creation and cleanup.
    
    Args:
        data: Array or tensor to plot
        title: Optional string for plot title
    """
    # Create figure explicitly
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Convert tensor if needed
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Create plot
    im = ax.imshow(data)
    plt.colorbar(im)
    if title:
        plt.title(title)
    
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
    
    # Clean up
    plt.close(fig)

def main():
    """Main entry point for the script."""
    args = parse_args()
    run_spectral_fpm_simulation(args.config_path)

if __name__ == "__main__":
    main() 