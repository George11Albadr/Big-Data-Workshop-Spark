import os
import time
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Ruta del archivo Parquet generado
PARQUET_PATH = "/home/jovyan/work/data/spotify/tracks_clustered.parquet"
if not os.path.exists(PARQUET_PATH):
    print(f"ERROR: No se encontró el archivo {PARQUET_PATH}. Ejecuta el bloque de escritura en tu notebook primero.")
    exit(1)

# Leer la data desde Parquet usando pyarrow y limitar a las primeras 5000 filas para optimizar la memoria
tracks_pd = pd.read_parquet(PARQUET_PATH, engine='pyarrow').head(5000)
print(f"DataFrame cargado correctamente con {len(tracks_pd)} filas.")

# Asegurarse de que las métricas de audio sean numéricas
metrics = ['danceability', 'energy', 'loudness', 'speechiness',
           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
for col in metrics:
    tracks_pd[col] = pd.to_numeric(tracks_pd[col], errors='coerce')

# Función para calcular la diversidad (distancia promedio entre canciones)
def compute_diversity(df):
    if df.shape[0] < 2:
        return 0.0
    X = df[metrics].to_numpy()
    return float(np.mean(pdist(X, metric='euclidean')))

# Crear la aplicación Dash con tema oscuro
app = Dash(__name__)

# Layout del Dashboard: se usan inputs globales en lugar de un slider por cada métrica
app.layout = html.Div(
    style={'backgroundColor': '#111111', 'color': '#FFFFFF', 'padding': '20px', 'fontFamily': 'Arial'},
    children=[
        html.H2("Dashboard de Recomendaciones Musicales", style={'textAlign': 'center'}),
        
        # Filtros globales
        html.Div([
            html.Div([
                html.Label("Popularidad Mínima:", style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='popularity-min', type='number', 
                    value=float(tracks_pd['popularity'].median()), min=0, max=100, step=1,
                    style={'width': '100%'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.Label("Danceability Mínima:", style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='danceability-min', type='number', 
                    value=float(tracks_pd['danceability'].median()), min=0, max=1, step=0.01,
                    style={'width': '100%'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.Label("Cluster:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='cluster-dropdown',
                    options=[{'label': f"Cluster {i}", 'value': i} for i in range(5)],
                    value=0,
                    clearable=False
                )
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'width': '100%', 'padding': '10px'}),
        
        # Contenedor de gráficos y tabla
        html.Div([
            # Tabla de canciones recomendadas
            html.Div([
                dash_table.DataTable(
                    id='rec-table',
                    columns=[
                        {'name': 'Canción', 'id': 'name'},
                        {'name': 'Artistas', 'id': 'artists'},
                        {'name': 'Año', 'id': 'release_year'},
                        {'name': 'Popularidad', 'id': 'popularity'},
                        {'name': 'Cluster', 'id': 'cluster'}
                    ],
                    data=[],  # Se llenará dinámicamente
                    page_size=10,
                    style_table={'maxHeight': '300px', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px', 'backgroundColor': '#111111', 'color': '#FFFFFF'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#222222', 'color': '#FFFFFF'}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
            
            # Gráficos: Barra, Pastel y Scatter
            html.Div([
                dcc.Graph(id='bar-chart'),
                dcc.Graph(id='pie-chart'),
                dcc.Graph(id='scatter-chart')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
        ]),
    ]
)

# Callback para actualizar la tabla y los gráficos según los filtros
@app.callback(
    [Output('rec-table', 'data'),
     Output('bar-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('scatter-chart', 'figure')],
    [Input('cluster-dropdown', 'value'),
     Input('popularity-min', 'value'),
     Input('danceability-min', 'value')]
)
def update_dashboard(selected_cluster, popularity_min, danceability_min):
    df = tracks_pd.copy()
    # Filtrar por cluster y umbrales de popularidad y danceability
    df = df[(df['cluster'] == selected_cluster) &
            (df['popularity'] >= popularity_min) &
            (df['danceability'] >= danceability_min)]
    
    # Datos para la tabla (máximo 50 registros)
    table_data = df[['name', 'artists', 'release_year', 'popularity', 'cluster']].head(50).to_dict('records')
    
    # Calcular métricas: diversidad y promedio de energía y loudness
    diversity_val = compute_diversity(df)
    avg_energy = df['energy'].mean() if not df.empty else 0.0
    avg_loudness = df['loudness'].mean() if not df.empty else 0.0

    metrics_df = pd.DataFrame({
        'Métrica': ['Diversidad', 'Energía Promedio', 'Loudness Promedio'],
        'Valor': [diversity_val, avg_energy, avg_loudness]
    })
    
    # Gráfico de barras
    bar_fig = px.bar(
        metrics_df, x='Métrica', y='Valor', color='Métrica',
        title="Métricas del Motor de Recomendación",
        color_discrete_sequence=['#1f77b4', '#2ca02c', '#ff7f0e']
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
    
    # Gráfico de pastel (para ver la contribución relativa de cada métrica)
    pie_fig = px.pie(
        metrics_df, names='Métrica', values='Valor',
        title="Distribución Relativa de Métricas",
        color_discrete_sequence=['#1f77b4', '#2ca02c', '#ff7f0e']
    )
    pie_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111111',
        plot_bgcolor='#111111'
    )
    
    # Gráfico scatter: comparar Energy vs Loudness (por ejemplo)
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
    
    return table_data, bar_fig, pie_fig, scatter_fig

if __name__ == '__main__':
    print("Iniciando el Dashboard de Recomendaciones Musicales en el puerto 8050...")
    time.sleep(3)
    print("La aplicación se ha iniciado. Visita http://127.0.0.1:8050 en tu navegador.")
    app.run(host="0.0.0.0", port=8050, debug=True)