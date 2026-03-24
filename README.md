# KMeans Clustering with Visualization

This project is an interactive web app for exploring how the KMeans clustering algorithm works step by step.

## What it does

The app generates a synthetic 2D dataset and lets you run KMeans clustering in an interactive way. Instead of only showing the final clusters, it helps you see the algorithm as it updates centroids over time.

## Main features

- Interactive clustering UI built with **Dash** and **Flask**
- Step-through execution of the KMeans algorithm
- Option to run KMeans all the way to convergence
- Multiple centroid initialization methods:
  - Random
  - Farthest First
  - KMeans++
  - Manual point selection
- Randomly generated datasets using `sklearn.datasets.make_blobs`
- Demo video included in the repository

## Files

- `app.py` - Dash app and UI logic
- `kmeans.py` - Custom implementation of KMeans clustering
- `requirements.txt` - Python dependencies
- `Makefile` - Helper commands for setup/run
- `kmeans demo.mp4` - Demo video of the app

## How it works

The app creates a synthetic dataset, plots the points, and allows the user to choose a number of clusters and an initialization strategy. The custom KMeans implementation then assigns points to the nearest centroid and updates centroids until convergence.

One useful detail is the **manual initialization** mode, where the user can click on the plot to choose starting centroids.

## Run locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   python app.py
   ```
3. Open the local Dash server in your browser.

## Why this repo is useful

This is a good teaching/demo project for understanding:

- how KMeans behaves under different initialization strategies
- how centroid placement affects convergence
- how to turn a machine learning concept into an interactive visualization