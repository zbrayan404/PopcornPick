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
movies_meta="../pickle/movies_meta.pkl"
movie="../datasets/movie.csv"
rating="../datasets/rating.csv"

# Check if all required files exist
if [ -f "$movies_meta" ] && [ -f "$movie" ] && [ -f "$rating" ]; then
    echo "All required files found. Proceeding to convert and run the notebook."

    # Ensure the pickle directory exists, if not create it
    pickle="../pickle"
    
    if [ ! -d "$pickle" ]; then
        echo "$pickle directory not found. Creating..."
        mkdir "$pickle"
    else
        echo "$pickle directory already exists."
    fi

    # Convert Jupyter notebook to Python script
    jupyter nbconvert --to script ../preprocess/userBasedPreprocess.ipynb

    # Run the converted Python script
    python ../preprocess/userBasedPreprocess.py

    # Clean up: remove the generated .py file if not needed
    rm ../preprocess/userBasedPreprocess.py
else
    echo "One or more required files (movies_meta.pkl, movie.csv, rating.csv) not found. Please ensure all datasets are available before running the script."
fi
