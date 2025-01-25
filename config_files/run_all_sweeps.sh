#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run sweeps sequentially
echo "Starting sweep 1 (180-300 LEDs)...NA=0.025"
bash "${SCRIPT_DIR}/run_led_sweep.sh"

echo "Starting sweep 2 (5-60 LEDs)...NA=0.1"
bash "${SCRIPT_DIR}/run_led_sweep2.sh"

# Add more sweeps as needed
# bash "${SCRIPT_DIR}/run_led_sweep3.sh"

echo "All sweeps completed!" 