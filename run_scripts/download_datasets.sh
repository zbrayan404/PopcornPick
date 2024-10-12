#!/bin/bash

# Check if kaggle.json exists
if [ ! -f kaggle.json ]; then
    echo "kaggle.json not found. Please place your Kaggle API credentials in run_scripts/kaggle.json"
    exit 1
fi

datasets="../datasets"

if [ ! -d "$datasets" ]; then
        echo "$datasets directory not found. Creating..."
        mkdir "$datasets"
    else
        echo "$datasets directory already exists."
    fi

# Ensure that the Kaggle API is installed
if ! command -v kaggle &> /dev/null
then
    echo "Kaggle CLI could not be found, installing..."
    pip install kaggle
fi

# Download the first dataset (e.g., movies dataset)
echo "Downloading the first dataset..."

kaggle datasets download -d akshaypawar7/millions-of-movies --unzip -p ../datasets

# Check if movies.csv exists and rename it to movies_meta.csv
if [ -f ../datasets/movies.csv ]; then
    mv ../datasets/movies.csv ../datasets/movies_meta.csv
    echo "movies.csv renamed to movies_meta.csv"
else
    echo "movies.csv not found, skipping rename"
fi

# Download the second dataset (e.g., dataset with multiple CSV files)
echo "Downloading the second dataset..."

kaggle datasets download -d grouplens/movielens-20m-dataset -p ../datasets

# Unzip only the required CSV files (movies.csv and ratings.csv) from the downloaded dataset
unzip -j ../datasets/*.zip 'movie.csv' 'rating.csv' -d ../datasets

# Remove the zip file after extraction
rm ../datasets/*.zip

echo "Both datasets have been downloaded and processed."

