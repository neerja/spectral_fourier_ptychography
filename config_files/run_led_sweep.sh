#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Loop through LED counts from x to x in steps of x
for num_leds in {20..200..20}; do
    echo "Running simulation with $num_leds LEDs..."
    
    # Create temporary config file with updated num_leds using Python
    python -c "
import json
with open('${SCRIPT_DIR}/config_num_led_sweep_na05.jsonc', 'r') as f:
    config = json.load(f)
config['led_array']['num_leds'] = ${num_leds}
with open('${SCRIPT_DIR}/config_${num_leds}.jsonc', 'w') as f:
    json.dump(config, f, indent=4)
"
    
    # Run the simulation with the temporary config
    python "${SCRIPT_DIR}/../run_fpm_simulation.py" "${SCRIPT_DIR}/config_${num_leds}.jsonc"
    
    # Clean up temporary config
    rm "${SCRIPT_DIR}/config_${num_leds}.jsonc"
done

echo "Completed all simulations!" 