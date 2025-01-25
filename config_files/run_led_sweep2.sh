#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# LOOP THROUGH TAU REG VALUES
for tau_reg in 0.0001 0.001 0.01; do
    # Loop through LED counts from x to x in steps of x
    for num_leds in {5..60..5}; do
        echo "Running simulation with $num_leds LEDs..."
        
        # Create temporary config file with updated num_leds using Python
        python -c "
    import json
    with open('${SCRIPT_DIR}/config_num_led_sweep_na01sparse.jsonc', 'r') as f:
        config = json.load(f)
    config['led_array']['num_leds'] = ${num_leds}
    config['reconstruction']['tau_reg'] = ${tau_reg}
    with open('${SCRIPT_DIR}/config_${num_leds}_${tau_reg}.jsonc', 'w') as f:
        json.dump(config, f, indent=4)
    "
        
        # Run the simulation with the temporary config
        python "${SCRIPT_DIR}/../run_fpm_simulation.py" "${SCRIPT_DIR}/config_${num_leds}_${tau_reg}.jsonc"
        
        # Clean up temporary config
        rm "${SCRIPT_DIR}/config_${num_leds}_${tau_reg}.jsonc"
    done
done

echo "Completed all simulations!" 