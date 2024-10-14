#!/bin/bash

# Example list of technique names
technique_names=("cf-gnnfeatures" "random-feat" "random" "cf-gnn" "ego" "cff")  # Replace with your actual technique names

# Loop over each technique name
for technique_name in "${technique_names[@]}"; do
    echo "Running with technique: $technique_name"
    
    # Run the Python command
    python main.py run_mode=sweep logger.mode=online technique=$technique_name dataset=planetoid dataset.name=cora
    

done
