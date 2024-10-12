#!/bin/bash

#Install required packages
pip install -r requirements.txt

# In run_scripts
cd run_scripts

# List of scripts to run
scripts=("download_datasets.sh" "run_notebook_pt1.sh" "run_notebook_pt2.sh")

# Loop through the scripts and run them in order
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"  # Execute the script
        
        # Check the exit status of the last executed command
        if [ $? -ne 0 ]; then
            echo "Execution of $script failed. Aborting execution."
            exit 1  # Exit the script if any script fails
        fi
    else
        echo "$script not found. Aborting execution."
        exit 1  # Exit the script if any script is not found
    fi
done

echo "All scripts have been processed successfully."

#Back 
cd ..

#Run app
streamlit run app.py