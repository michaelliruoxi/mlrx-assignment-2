import dash
import numpy as np
from flask import Flask
from dash import Dash, dcc, html, Input, Output, State
from sklearn.datasets import make_blobs
from kmeans import KMeansCustom

# Initialize Flask server
server = Flask(__name__)

# Initialize Dash app
app = Dash(__name__, server=server, external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"])

# Global variables
X, y = make_blobs(n_samples=300, centers=1, cluster_std=5.0, center_box=(-10.0, 10.0), random_state=None)  # Fully spread-out dataset
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
        X, _ = make_blobs(n_samples=300, centers=1, cluster_std=5.0, center_box=(-10.0, 10.0), random_state=None)
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
    if click_data and len(manual_centroids) < n_clusters:
        # Extract the clicked point
        x_click = click_data['points'][0]['x']
        y_click = click_data['points'][0]['y']
        
        # Check if the clicked point is valid and append it to manual_centroids
        manual_centroids.append([x_click, y_click])
        print(f"Manual centroids so far: {manual_centroids}")
    
        # Once enough centroids are selected, initialize KMeans
        if len(manual_centroids) == n_clusters and kmeans is None:
            print(f"Initializing KMeans with {n_clusters} manual centroids: {manual_centroids}")
            kmeans = KMeansCustom(n_clusters=n_clusters, init_method='manual', manual_centroids=manual_centroids)
            kmeans.centroids = np.array(manual_centroids)
        


    # Step through KMeans after manual initialization or other methods
    if triggered_input == 'step_kmeans' and not converged:
        if kmeans is None or kmeans.centroids is None:
            if init_method == 'manual' and len(manual_centroids) < n_clusters:
                convergence_message = f"Please select {n_clusters} centroids on the plot."
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
                convergence_message = f"Please select {n_clusters} centroids on the plot."
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
            }
        ],
        'layout': {
            'title': 'KMeans Clustering Data',
            'xaxis': {'title': 'X-axis'},
            'yaxis': {'title': 'Y-axis'},
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
