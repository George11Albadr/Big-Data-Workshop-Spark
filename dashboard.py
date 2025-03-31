import os
import time
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------
# 1) LOAD DATA FROM PARQUET
# ----------------------------------------------------------------
PARQUET_PATH = "/home/jovyan/work/data/spotify/tracks_clustered.parquet"
if not os.path.exists(PARQUET_PATH):
    print(f"ERROR: No se encontró el archivo {PARQUET_PATH}. Ejecuta el bloque de escritura en tu notebook primero.")
    exit(1)

# Load up to 5000 rows to limit memory usage
tracks_pd = pd.read_parquet(PARQUET_PATH, engine='pyarrow').head(5000)
print(f"DataFrame cargado correctamente con {len(tracks_pd)} filas.")

# Convert columns to numeric if needed
metrics = [
    'danceability','energy','loudness','speechiness',
    'acousticness','instrumentalness','liveness','valence','tempo'
]
for col in metrics:
    tracks_pd[col] = pd.to_numeric(tracks_pd[col], errors='coerce')

# ----------------------------------------------------------------
# 2) DEFINE HELPER FUNCTIONS
# ----------------------------------------------------------------
def compute_diversity(df):
    """
    Diversidad = distancia promedio (euclidiana) entre canciones en df[metrics].
    """
    if df.shape[0] < 2:
        return 0.0
    X = df[metrics].to_numpy()
    return float(np.mean(pdist(X, metric='euclidean')))

# ----------------------------------------------------------------
# 3) CREATE DASH APP
# ----------------------------------------------------------------
app = Dash(__name__)

app.layout = html.Div(
    style={
        'backgroundColor': '#111111',
        'color': '#FFFFFF',
        'padding': '20px',
        'fontFamily': 'Arial'
    },
    children=[

        html.H1("Dashboard de Recomendaciones Musicales",
                style={'textAlign': 'center', 'marginBottom': '20px'}),

        # -------------------------
        # TOP FILTERS
        # -------------------------
        html.Div([
            html.Div([
                html.Label("Popularidad mínima (0-100):", style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='popularity-min', type='number',
                    value=float(tracks_pd['popularity'].median()), 
                    min=0, max=100, step=1,
                    style={'width': '100%'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                html.Label("Danceability mínima (0-1):", style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='danceability-min', type='number',
                    value=float(tracks_pd['danceability'].median()), 
                    min=0, max=1, step=0.01,
                    style={'width': '100%'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                html.Label("Seleccionar Cluster:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='cluster-dropdown',
                    options=[{'label': f"Cluster {i}", 'value': i} for i in range(5)],
                    value=0,
                    clearable=False,
                    style={'color': '#000000'}  # so text is visible in the dropdown
                )
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
        ]),

        # -------------------------
        # TABLE + METRICS GRAPHS
        # -------------------------
        html.Div([
            # Table on the left
            html.Div([
                html.H3("Canciones Recomendadas", style={'textAlign': 'left'}),
                dash_table.DataTable(
                    id='rec-table',
                    columns=[
                        {'name': 'Canción', 'id': 'name'},
                        {'name': 'Artistas', 'id': 'artists'},
                        {'name': 'Año', 'id': 'release_year'},
                        {'name': 'Popularidad', 'id': 'popularity'},
                        {'name': 'Cluster', 'id': 'cluster'}
                    ],
                    data=[],
                    page_size=10,
                    style_table={'maxHeight': '350px', 'overflowY': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '5px',
                        'backgroundColor': '#111111',
                        'color': '#FFFFFF'
                    },
                    style_header={
                        'fontWeight': 'bold',
                        'backgroundColor': '#222222',
                        'color': '#FFFFFF'
                    }
                )
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            # Right column with Bar + Pie + new chart
            html.Div([
                # 1) Bar chart: metrics
                dcc.Graph(id='bar-chart'),

                # 2) Pie chart: distribution of metrics
                dcc.Graph(id='pie-chart'),

                # 3) Another chart: let's do a scatter (Energy vs Loudness) or something else
                dcc.Graph(id='scatter-chart')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'width': '100%', 'padding': '10px'}),

        # -------------------------
        # ADDITIONAL BOTTOM GRAPHS
        # -------------------------
        html.Div([
            # 4) Hist for valence
            dcc.Graph(id='hist-valence', style={'display': 'inline-block', 'width': '48%'}),

            # 5) Box plot for acousticness vs. popularity
            dcc.Graph(id='box-acousticness', style={'display': 'inline-block', 'width': '48%'})
        ], style={'width': '100%', 'padding': '10px'})
    ]
)

# ----------------------------------------------------------------
# 4) CALLBACKS
# ----------------------------------------------------------------
@app.callback(
    [
        Output('rec-table', 'data'),
        Output('bar-chart', 'figure'),
        Output('pie-chart', 'figure'),
        Output('scatter-chart', 'figure'),
        Output('hist-valence', 'figure'),
        Output('box-acousticness', 'figure')
    ],
    [
        Input('cluster-dropdown', 'value'),
        Input('popularity-min', 'value'),
        Input('danceability-min', 'value')
    ]
)
def update_dashboard(selected_cluster, popularity_min, danceability_min):
    df = tracks_pd.copy()

    # Filter data
    df = df[
        (df['cluster'] == selected_cluster) &
        (df['popularity'] >= popularity_min) &
        (df['danceability'] >= danceability_min)
    ]

    # TABLE
    table_data = df[['name','artists','release_year','popularity','cluster']].head(50).to_dict('records')

    # METRICS
    diversity_val = compute_diversity(df)
    avg_energy = df['energy'].mean() if not df.empty else 0.0
    avg_loudness = df['loudness'].mean() if not df.empty else 0.0

    # We could add more metrics if desired
    metrics_df = pd.DataFrame({
        'Métrica': ['Diversidad', 'Energía Promedio', 'Loudness Promedio'],
        'Valor': [diversity_val, avg_energy, avg_loudness]
    })

    # 1) Bar Chart
    bar_fig = px.bar(
        metrics_df, x='Métrica', y='Valor', color='Métrica',
        title="Métricas del Motor de Recomendación",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    bar_fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    bar_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        yaxis_title="Valor",
        xaxis_title="Métrica",
        showlegend=False
    )

    # 2) Pie Chart
    # We'll just show the relative distribution among the three metrics
    pie_fig = px.pie(
        metrics_df, names='Métrica', values='Valor',
        title="Distribución Relativa de Métricas",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    pie_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111111',
        plot_bgcolor='#111111'
    )

    # 3) Scatter Chart (Energy vs Loudness)
    scatter_fig = px.scatter(
        df, x='energy', y='loudness', color='popularity',
        title="Relación Energy vs Loudness",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    scatter_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        xaxis_title="Energy",
        yaxis_title="Loudness"
    )

    # 4) Histogram for valence
    hist_valence = px.histogram(
        df, x='valence', nbins=20,
        title="Distribución de Valence",
        color_discrete_sequence=['#2ca02c']
    )
    hist_valence.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        xaxis_title="Valence",
        yaxis_title="Frecuencia"
    )

    # 5) Box plot for acousticness vs popularity (binned)
    # Create a 'pop_bin' to categorize popularity
    if not df.empty:
        df['pop_bin'] = pd.cut(df['popularity'], bins=[0, 25, 50, 75, 100],
                            labels=['0-25','25-50','50-75','75-100'])
    else:
        df['pop_bin'] = None

    box_acoustic = px.box(
        df, x='pop_bin', y='acousticness',
        title="Acousticness por Rango de Popularidad",
        color='pop_bin',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    box_acoustic.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        xaxis_title="Popularidad (bins)",
        yaxis_title="Acousticness",
        showlegend=False
    )

    return table_data, bar_fig, pie_fig, scatter_fig, hist_valence, box_acoustic

# ----------------------------------------------------------------
# 5) RUN SERVER
# ----------------------------------------------------------------
if __name__ == '__main__':
    print("Iniciando el Dashboard de Recomendaciones Musicales en el puerto 8050...")
    time.sleep(3)
    print("La aplicación se ha iniciado. Visita http://127.0.0.1:8050 en tu navegador.")
    # Listen on all interfaces so Docker can map the port
    app.run(host="0.0.0.0", port=8050, debug=True)