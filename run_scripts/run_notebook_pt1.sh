#!/bin/bash

# Ensure nbconvert and ipython are installed
if ! pip show nbconvert > /dev/null 2>&1; then
    echo "nbconvert not found. Installing nbconvert..."
    pip install nbconvert
else
    echo "nbconvert is already installed."
fi

if ! pip show ipython > /dev/null 2>&1; then
    echo "ipython not found. Installing ipython..."
    pip install ipython
else
    echo "ipython is already installed."
fi

# File path to check
movies_meta="../datasets/movies_meta.csv"

# Check if the file exists
if [ -f "$movies_meta" ]; then
    echo "$movies_meta found. Proceeding to convert and run the notebook."

    # Ensure the pickle directory exists, if not create it
    pickle="../pickle"
    
    if [ ! -d "$pickle" ]; then
        echo "$pickle directory not found. Creating..."
        mkdir "$pickle"
    else
        echo "$pickle directory already exists."
    fi

    # Convert Jupyter notebook to Python script
    jupyter nbconvert --to script ../preprocess/contentBasedPreprocess.ipynb

    # Run the converted Python script
    python ../preprocess/contentBasedPreprocess.py

    # Clean up: remove the generated .py file if not needed
    rm ../preprocess/contentBasedPreprocess.py
else
    echo "$movies_meta not found. Please ensure the dataset is available before running the script."
fi
