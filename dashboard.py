import os
import time
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px

# ----------------------------------------------------------------
# 1) CARGA DE DATA DESDE PARQUET
# ----------------------------------------------------------------
PARQUET_PATH = "/home/jovyan/work/data/spotify/tracks_clustered.parquet"
if not os.path.exists(PARQUET_PATH):
    print(f"ERROR: No se encontró el archivo {PARQUET_PATH}. Ejecuta el bloque de escritura en tu notebook primero.")
    exit(1)

# Limitar a las primeras 5000 filas para optimizar memoria
tracks_pd = pd.read_parquet(PARQUET_PATH, engine='pyarrow').head(5000)
print(f"DataFrame cargado correctamente con {len(tracks_pd)} filas.")

# Definir las métricas (columnas numéricas de audio)
metrics = ['danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Asegurar que las columnas sean numéricas
for col in metrics:
    tracks_pd[col] = pd.to_numeric(tracks_pd[col], errors='coerce')

# Crear columna 'decade' a partir de 'release_year' si existe, de lo contrario crearla con NaN
if 'release_year' in tracks_pd.columns:
    tracks_pd['decade'] = (tracks_pd['release_year'] // 10) * 10
else:
    tracks_pd['decade'] = np.nan

# Crear columna 'pop_bin' para el box plot (bins de popularidad)
if 'popularity' in tracks_pd.columns:
    tracks_pd['pop_bin'] = pd.cut(tracks_pd['popularity'], bins=[0, 25, 50, 75, 100],
                                labels=['0-25', '25-50', '50-75', '75-100'])
else:
    tracks_pd['pop_bin'] = None

# ----------------------------------------------------------------
# 2) FUNCIÓN AUXILIAR: CÁLCULO DE DIVERSIDAD
# ----------------------------------------------------------------
def compute_diversity(df):
    if df.shape[0] < 2:
        return 0.0
    X = df[metrics].to_numpy()
    return float(np.mean(pdist(X, metric='euclidean')))

# ----------------------------------------------------------------
# 3) CREACIÓN DE LA APLICACIÓN DASH CON TEMA OSCURO Y LAYOUT FLEX
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
        html.H2("Dashboard de Recomendaciones Musicales", style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Contenedor de filtros (Flex layout)
        html.Div(
            style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'},
            children=[
                html.Div([
                    html.Label("Cluster:", style={'fontWeight': 'bold', 'fontSize': '12px'}),
                    dcc.Dropdown(
                        id='cluster-dropdown',
                        options=[{'label': f"Cluster {i}", 'value': i} for i in range(5)],
                        value=0,
                        clearable=False,
                        style={'width': '100px', 'fontSize': '12px'}
                    )
                ], style={'flex': '1', 'minWidth': '120px', 'padding': '5px'}),
                html.Div([
                    html.Label("Popularidad Mínima:", style={'fontWeight': 'bold', 'fontSize': '12px'}),
                    dcc.Input(
                        id='popularity-min',
                        type='number',
                        value=float(tracks_pd['popularity'].median()),
                        min=0, max=100, step=1,
                        style={'width': '60px', 'fontSize': '12px'}
                    )
                ], style={'flex': '1', 'minWidth': '120px', 'padding': '5px'}),
                html.Div([
                    html.Label("Danceability Mínima:", style={'fontWeight': 'bold', 'fontSize': '12px'}),
                    dcc.Input(
                        id='danceability-min',
                        type='number',
                        value=float(tracks_pd['danceability'].median()),
                        min=0, max=1, step=0.01,
                        style={'width': '60px', 'fontSize': '12px'}
                    )
                ], style={'flex': '1', 'minWidth': '120px', 'padding': '5px'}),
                html.Div([
                    html.Label("Década:", style={'fontWeight': 'bold', 'fontSize': '12px'}),
                    dcc.Dropdown(
                        id='decade-dropdown',
                        options=[{'label': str(dec), 'value': dec} for dec in sorted(tracks_pd['decade'].dropna().unique())],
                        placeholder="Todas",
                        multi=False,
                        clearable=True,
                        style={'width': '100px', 'fontSize': '12px'}
                    )
                ], style={'flex': '1', 'minWidth': '120px', 'padding': '5px'}),
                html.Div([
                    html.Label("Métrica Histograma:", style={'fontWeight': 'bold', 'fontSize': '12px'}),
                    dcc.Dropdown(
                        id='histogram-metric',
                        options=[{'label': m.capitalize(), 'value': m} for m in metrics],
                        value='valence',
                        clearable=False,
                        style={'width': '140px', 'fontSize': '12px'}
                    )
                ], style={'flex': '1', 'minWidth': '140px', 'padding': '5px'}),
                html.Div([
                    html.Label("Métrica Box Plot:", style={'fontWeight': 'bold', 'fontSize': '12px'}),
                    dcc.Dropdown(
                        id='boxplot-metric',
                        options=[{'label': m.capitalize(), 'value': m} for m in metrics],
                        value='acousticness',
                        clearable=False,
                        style={'width': '140px', 'fontSize': '12px'}
                    )
                ], style={'flex': '1', 'minWidth': '140px', 'padding': '5px'})
            ]
        ),
        
        html.Br(),
        
        # Contenedor de la tabla y gráficos
        html.Div(
            style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'},
            children=[
                # Tabla de canciones recomendadas (ocupando 100% de ancho)
                html.Div([
                    html.H3("Canciones Recomendadas", style={'textAlign': 'left', 'fontSize': '14px'}),
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
                        style_table={'maxHeight': '300px', 'overflowY': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '5px',
                            'backgroundColor': '#111111',
                            'color': '#FFFFFF',
                            'fontSize': '12px'
                        },
                        style_header={
                            'fontWeight': 'bold',
                            'backgroundColor': '#222222',
                            'color': '#FFFFFF',
                            'fontSize': '12px'
                        }
                    )
                ], style={'flex': '1 1 100%', 'padding': '10px'}),
                
                # Contenedor de gráficos organizados en filas (usando Flex)
                html.Div(
                    style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'},
                    children=[
                        # Primera fila de gráficos
                        html.Div([dcc.Graph(id='bar-chart')], style={'flex': '1 1 48%', 'padding': '10px'}),
                        html.Div([dcc.Graph(id='pie-chart')], style={'flex': '1 1 48%', 'padding': '10px'}),
                        # Segunda fila de gráficos
                        html.Div([dcc.Graph(id='scatter-chart')], style={'flex': '1 1 48%', 'padding': '10px'}),
                        html.Div([dcc.Graph(id='histogram-chart')], style={'flex': '1 1 48%', 'padding': '10px'}),
                        # Tercera fila: gráfico adicional (box plot)
                        html.Div([dcc.Graph(id='boxplot-chart')], style={'flex': '1 1 48%', 'padding': '10px'})
                    ]
                )
            ]
        )
    ]
)

# ----------------------------------------------------------------
# 4) CALLBACKS PARA ACTUALIZAR EL DASHBOARD
# ----------------------------------------------------------------
@app.callback(
    [Output('rec-table', 'data'),
    Output('bar-chart', 'figure'),
    Output('pie-chart', 'figure'),
    Output('scatter-chart', 'figure'),
    Output('histogram-chart', 'figure'),
    Output('boxplot-chart', 'figure')],
    [Input('cluster-dropdown', 'value'),
    Input('popularity-min', 'value'),
    Input('danceability-min', 'value'),
    Input('decade-dropdown', 'value'),
    Input('histogram-metric', 'value'),
    Input('boxplot-metric', 'value')]
)
def update_dashboard(selected_cluster, popularity_min, danceability_min, selected_decade, selected_hist_metric, selected_box_metric):
    df = tracks_pd.copy()
    # Filtrar por cluster, popularidad y danceability
    df = df[(df['cluster'] == selected_cluster) &
            (df['popularity'] >= popularity_min) &
            (df['danceability'] >= danceability_min)]
    # Filtrar por década si se selecciona
    if selected_decade is not None:
        df = df[df['decade'] == selected_decade]
    
    # Datos para la tabla (máximo 50 registros)
    table_data = df[['name', 'artists', 'release_year', 'popularity', 'cluster']].head(50).to_dict('records')
    
    # Calcular métricas generales
    diversity_val = compute_diversity(df)
    avg_energy = df['energy'].mean() if not df.empty else 0.0
    avg_loudness = df['loudness'].mean() if not df.empty else 0.0
    avg_speechiness = df['speechiness'].mean() if not df.empty else 0.0
    avg_acousticness = df['acousticness'].mean() if not df.empty else 0.0
    avg_instrumentalness = df['instrumentalness'].mean() if not df.empty else 0.0
    avg_liveness = df['liveness'].mean() if not df.empty else 0.0
    avg_valence = df['valence'].mean() if not df.empty else 0.0
    avg_tempo = df['tempo'].mean() if not df.empty else 0.0

    metrics_df = pd.DataFrame({
        'Métrica': [
            'Diversidad', 'Energía Promedio', 'Loudness Promedio', 'Speechiness Promedio',
            'Acousticness Promedio', 'Instrumentalness Promedio', 'Liveness Promedio',
            'Valence Promedio', 'Tempo Promedio'
        ],
        'Valor': [
            diversity_val, avg_energy, avg_loudness, avg_speechiness,
            avg_acousticness, avg_instrumentalness, avg_liveness,
            avg_valence, avg_tempo
        ]
    })
    
    # Gráfico de barras de métricas generales
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
        showlegend=False,
        font=dict(size=10)
    )
    
    # Gráfico de pastel: distribución relativa de las métricas (usa los valores generales)
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
    
    # Scatter: Energy vs Loudness
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
    
    # Histograma para la métrica seleccionada
    hist_chart = px.histogram(
        df, x=selected_hist_metric, nbins=20,
        title=f"Distribución de {selected_hist_metric.capitalize()}",
        color_discrete_sequence=['#2ca02c']
    )
    hist_chart.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        xaxis_title=selected_hist_metric.capitalize(),
        yaxis_title="Frecuencia"
    )
    
    # Box plot: Métrica seleccionada por rango de popularidad (usando 'pop_bin')
    box_fig = px.box(
        df, x='pop_bin', y=selected_box_metric,
        title=f"{selected_box_metric.capitalize()} por Rango de Popularidad",
        color='pop_bin',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    box_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        xaxis_title="Popularidad (bins)",
        yaxis_title=selected_box_metric.capitalize(),
        showlegend=False
    )
    
    return table_data, bar_fig, pie_fig, scatter_fig, hist_chart, box_fig

# ----------------------------------------------------------------
# 5) EJECUCIÓN DEL SERVIDOR
# ----------------------------------------------------------------
if __name__ == '__main__':
    print("Iniciando el Dashboard de Recomendaciones Musicales en el puerto 8050...")
    time.sleep(3)
    print("La aplicación se ha iniciado. Visita http://127.0.0.1:8050 en tu navegador.")
    app.run(host="0.0.0.0", port=8050, debug=True)