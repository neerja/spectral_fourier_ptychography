#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if a config file was provided as an argument
if [ $# -eq 0 ]; then
    CONFIG_FILE="${SCRIPT_DIR}/config_na01_spectral.jsonc"
else
    CONFIG_FILE="$1"
fi

echo "Using config file: $CONFIG_FILE"

# Define tau_reg values using scientific notation
tau_values=("1e-4" "1e-5" "1e-3")

for tau in "${tau_values[@]}"; do
    echo "Running simulation with tau_reg = $tau..."
    
    # Create temporary config file with updated tau_reg using Python
    python -c "
import json
with open('${CONFIG_FILE}', 'r') as f:
    config = json.load(f)
config['reconstruction']['tau_reg'] = float('${tau}')
with open('${CONFIG_FILE%.jsonc}_tau_${tau}.jsonc', 'w') as f:
    json.dump(config, f, indent=4)
"
    # Run the simulation with the temporary config
    python "${SCRIPT_DIR}/../run_fpm_simulation.py" "${CONFIG_FILE%.jsonc}_tau_${tau}.jsonc"
    
    # Clean up temporary config
    rm "${CONFIG_FILE%.jsonc}_tau_${tau}.jsonc"
done

echo "Completed all simulations!" 