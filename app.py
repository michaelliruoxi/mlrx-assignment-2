import dash
import numpy as np
from flask import Flask
from dash import Dash, dcc, html, Input, Output, State
from sklearn.datasets import make_blobs
from kmeans import KMeansCustom
from scipy.spatial import cKDTree

# Initialize Flask server
server = Flask(__name__)

# Initialize Dash app
app = Dash(__name__, server=server, external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"])

# Global variables
# Randomize parameters
n_samples = np.random.randint(200, 500)  # Random number of samples
n_centers = np.random.randint(1, 10)  # Random number of centers (clusters)
cluster_std = np.random.uniform(1.0, 2.0)  # Random standard deviation for cluster spread
center_box = (np.random.uniform(-20.0, 0.0), np.random.uniform(0.0, 20.0))  # Random center range


# Generate random dataset
X, y = make_blobs(n_samples=n_samples, centers=n_centers, cluster_std=cluster_std, center_box=center_box)

def create_invisible_grid(X, num_points=50, threshold=0.1):
    # Create a grid based on the min and max of the X data
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    # Create linearly spaced points in the x and y direction
    x_grid = np.linspace(x_min, x_max, num_points)
    y_grid = np.linspace(y_min, y_max, num_points)

    # Create a meshgrid for all combinations of x and y points
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Create grid points as (x, y) pairs
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Remove points that are too close to actual data points
    # Use a KDTree to efficiently find close neighbors
    tree = cKDTree(X)  # Build a KDTree of the data points
    distances, _ = tree.query(grid_points, distance_upper_bound=threshold)  # Query points within a threshold distance

    # Only keep grid points that are farther than the threshold distance from data points
    mask = distances > threshold  # Boolean mask: True if point is far enough
    filtered_grid_points = grid_points[mask]

    return filtered_grid_points

kmeans = None  # No centroids initially
current_step = 0
converged = False

# App layout
app.layout = html.Div([
    html.H1("KMeans Clustering Algorithm", style={'textAlign': 'center'}),

    html.Div([
        html.Label('Number of Clusters (k):'),
        dcc.Input(id='num_clusters', type='number', value=3, min=2, max=10),
    ]),

    html.Div([
        html.Label('Initialization Method:'),
        dcc.Dropdown(
            id='init_method',
            options=[
                {'label': 'Random', 'value': 'random'},
                {'label': 'Farthest First', 'value': 'farthest_first'},
                {'label': 'KMeans++', 'value': 'kmeans++'},
                {'label': 'Manual', 'value': 'manual'},
            ],
            value='random',
        ),
    ]),

    html.Button('Step Through KMeans', id='step_kmeans', n_clicks=0),
    html.Button('Run to Convergence', id='run_convergence', n_clicks=0),
    html.Button('Generate New Dataset', id='new_dataset', n_clicks=0),
    html.Button('Reset Algorithm', id='reset_kmeans', n_clicks=0),

    # Message Div to show convergence status
    html.Div(id='convergence_message', style={'color': 'red', 'font-size': '20px'}),

    dcc.Graph(id='cluster_graph', config={'displayModeBar': False}, clear_on_unhover=True),

    # Hidden div to store manual centroids
    dcc.Store(id='manual_centroids_storage', data=[])
], style={'width': '50%', 'display': 'inline-block', 'padding': '20px'})


# Update plot on various button clicks and capture click events for manual selection
@app.callback(
    [Output('cluster_graph', 'figure'),
     Output('convergence_message', 'children'),
     Output('manual_centroids_storage', 'data')],
    [
        Input('num_clusters', 'value'),
        Input('init_method', 'value'),
        Input('step_kmeans', 'n_clicks'),
        Input('run_convergence', 'n_clicks'),
        Input('new_dataset', 'n_clicks'),
        Input('reset_kmeans', 'n_clicks'),
        Input('cluster_graph', 'clickData')  # Capture click events for manual selection
    ],
    [State('manual_centroids_storage', 'data')],
    prevent_initial_call=False  # This allows the first call to initialize the plot with default data
)
def update_plot(n_clusters, init_method, step_clicks, convergence_clicks, new_dataset_clicks, reset_clicks, click_data, manual_centroids):
    global X, kmeans, current_step, converged

    # Detect which input triggered the callback
    ctx = dash.callback_context
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Message to display when converged
    convergence_message = ''

    # Generate new dataset
    if triggered_input == 'new_dataset':
        # Randomize parameters
        n_samples = np.random.randint(200, 500)  # Random number of samples
        n_centers = np.random.randint(1, 10)  # Random number of centers (clusters)
        cluster_std = np.random.uniform(1.0, 2.0)  # Random standard deviation for cluster spread
        center_box = (np.random.uniform(-20.0, 0.0), np.random.uniform(0.0, 20.0))  # Random center range


        # Generate random dataset
        X, y = make_blobs(n_samples=n_samples, centers=n_centers, cluster_std=cluster_std, center_box=center_box)
        kmeans = None  # Clear centroids initially
        manual_centroids = []  # Clear manual centroids
        current_step = 0  # Reset step counter
        converged = False  # Reset convergence status
        convergence_message = ''  # Clear the convergence message
        click_data = {}

    # Reset the KMeans model, keeping the current dataset
    elif triggered_input == 'reset_kmeans':
        kmeans = None  # Clear centroids initially
        manual_centroids = []  # Clear manual centroids
        current_step = 0  # Reset step counter
        converged = False  # Reset convergence status
        convergence_message = ''  # Clear the convergence message
        click_data = {}

    

    # Prevent further graph updates when KMeans has converged
    if converged:
        if triggered_input in ['step_kmeans', 'run_convergence']:
            convergence_message = "KMeans has converged!"
            return dash.no_update, convergence_message, manual_centroids
    
    # Allow user to select centroids up to the number of clusters specified by `n_clusters`
    if click_data and len(manual_centroids) < n_clusters and current_step < 1:
        # Extract the clicked point
        x_click = click_data['points'][0]['x']
        y_click = click_data['points'][0]['y']
        
        # Check if the clicked point is valid and append it to manual_centroids
        manual_centroids.append([x_click, y_click])
        print(f"Manual centroids so far: {manual_centroids}")
        print(len(manual_centroids))
        print(n_clusters)
        # Once enough centroids are selected, initialize KMeans
        if len(manual_centroids) == n_clusters and kmeans is None:
            print(f"Initializing KMeans with {n_clusters} manual centroids: {manual_centroids}")
            kmeans = KMeansCustom(n_clusters=n_clusters, init_method='manual', manual_centroids=manual_centroids)
            kmeans.centroids = np.array(manual_centroids)
        
    invisible_grid = create_invisible_grid(X, num_points=100)

    # Step through KMeans after manual initialization or other methods
    if triggered_input == 'step_kmeans' and not converged:
        if kmeans is None or kmeans.centroids is None:
            if init_method == 'manual' and len(manual_centroids) < n_clusters:
                convergence_message = f"Please select {n_clusters} centroids on the plot. Press Reset Algorithm to try again."
                return dash.no_update, convergence_message, manual_centroids
            kmeans = KMeansCustom(n_clusters=n_clusters, init_method=init_method)
            if init_method == 'manual':
                kmeans.centroids = np.array(manual_centroids)
            kmeans.initialize_centroids(X)  # Initialize centroids (manual or other method)

        if current_step > 0:
            # Perform one iteration of KMeans
            closest_centroids = kmeans._assign_clusters(X)
            new_centroids = np.array([X[closest_centroids == k].mean(axis=0) for k in range(n_clusters)])

            # Check if centroids have changed
            if np.all(kmeans.centroids == new_centroids):
                converged = True  # Mark as converged if centroids do not change
                convergence_message = "KMeans has converged!"
            else:
                kmeans.centroids = new_centroids  # Update centroids for the next step
                current_step += 1  # Increment step count
        else: 
            current_step += 1

    # Run KMeans until convergence when the button is pressed
    elif triggered_input == 'run_convergence':
        if kmeans is None:
            if init_method == 'manual' and len(manual_centroids) < n_clusters:
                convergence_message = f"Please select {n_clusters} centroids on the plot. Press Reset Algorithm to try again."
                return dash.no_update, convergence_message, manual_centroids
            kmeans = KMeansCustom(n_clusters=n_clusters, init_method=init_method)
            if init_method == 'manual':
                kmeans.centroids = np.array(manual_centroids)  # Use manually selected centroids

        kmeans.fit(X)  # Run until convergence
        converged = True  # Mark as converged
        convergence_message = "KMeans has converged!"


    # Assign clusters after each iteration (or full convergence)
    if kmeans and (current_step != 0 or converged == True): 
        closest_centroids = kmeans._assign_clusters(X)
        closest_centroids = kmeans._assign_clusters(X)
        # Create color mapping for each cluster
        colors = ['green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'pink', 'brown']
        point_colors = [colors[label % len(colors)] for label in closest_centroids]
    else:
        # If no centroids, just display all points with the same color (initial state)
        point_colors = 'blue'

    # Create plot to display
    figure = {
        'data': [
            {
                'x': X[:, 0],
                'y': X[:, 1],
                'mode': 'markers',
                'marker': {'color': point_colors},
                'name': 'Data Points'
            },{
                'x': invisible_grid[:, 0],
                'y': invisible_grid[:, 1],
                'mode': 'markers',
                'marker': {'color': 'rgba(0,0,0,0)'},  # Make the grid points invisible
                'name': '',
                'opacity': 0  # Ensure they are invisible
            }
        ],
        'layout': {
            'title': 'KMeans Clustering Data',
            'xaxis': {'title': 'X-axis'},
            'yaxis': {'title': 'Y-axis'},
            'hovermode': 'closest',
            'autosize': True
        }
    }

    if (current_step == 1 or converged == True) and init_method == "manual":
        # Clear the figure data
        manual_centroids = []
        figure['data'] = [
            {
                'x': X[:, 0],
                'y': X[:, 1],
                'mode': 'markers',
                'marker': {'color': point_colors},  # Reset the color of all data points
                'name': 'Data Points'
            }
        ]
        print("Graph has been reset after the first step.")

    # If manual centroids are selected, display them as red crosses
    if manual_centroids and init_method == 'manual' and current_step<1:
        manual_centroids = np.array(manual_centroids)
        figure['data'].append({
            'x': manual_centroids[:, 0],
            'y': manual_centroids[:, 1],
            'mode': 'markers',
            'marker': {'symbol': 'x', 'size': 12, 'color': 'red'},
            'name': 'Centroids'
        })

    # If centroids are available, add them to the plot
    if kmeans:
        figure['data'].append({
            'x': kmeans.centroids[:, 0],
            'y': kmeans.centroids[:, 1],
            'mode': 'markers',
            'marker': {'symbol': 'x', 'size': 12, 'color': 'red'},
            'name': 'Centroids'
        })
    
    return figure, convergence_message, manual_centroids

if __name__ == '__main__':
    app.run_server(host='localhost', port=3000)
