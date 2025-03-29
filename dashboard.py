import os
import time
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px

# Ruta del archivo Parquet generado (asegúrate de que tracks_clustered.parquet existe)
PARQUET_PATH = "/home/jovyan/work/data/spotify/tracks_clustered.parquet"
if not os.path.exists(PARQUET_PATH):
    print(f"ERROR: No se encontró el archivo {PARQUET_PATH}. Asegúrate de haberlo guardado con Spark.")
    exit(1)

# Cargar el DataFrame de Pandas
tracks_pd = pd.read_parquet(PARQUET_PATH)
print(f"DataFrame cargado correctamente con {len(tracks_pd)} filas.")

# Lista de métricas de audio a filtrar
metrics = ['danceability', 'energy', 'loudness', 'speechiness',
           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Función para calcular diversidad (distancia promedio entre canciones)
def compute_diversity(df):
    features_cols = metrics
    # Asegurarse de que las columnas estén presentes
    features_cols = [col for col in features_cols if col in df.columns]
    if df.shape[0] < 2 or not features_cols:
        return 0.0
    X = df[features_cols].to_numpy()
    return float(np.mean(pdist(X, metric='euclidean')))

# Generar controles (sliders) para cada métrica de audio
# Se obtienen los valores mínimo y máximo a partir del DataFrame
def create_slider(metric):
    min_val = float(tracks_pd[metric].min())
    max_val = float(tracks_pd[metric].max())
    # Para la mayoría de métricas (entre 0 y 1) se puede usar step 0.05, para tempo se usa 1
    step = 1 if metric == 'tempo' else 0.05
    return html.Div([
        html.Label(f"{metric.capitalize()} mínima ({min_val:.2f} - {max_val:.2f}):", style={'marginRight': '10px'}),
        dcc.Slider(
            id=f"slider-{metric}",
            min=min_val,
            max=max_val,
            step=step,
            value=min_val,
            marks={round(v,2): str(round(v,2)) for v in np.linspace(min_val, max_val, num=5)}
        )
    ], style={'padding': '10px'})

# Crear lista de controles para cada métrica
slider_controls = [create_slider(metric) for metric in metrics]

# Crear la aplicación Dash con tema oscuro
app = Dash(__name__)
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#FFFFFF', 'padding': '20px'}, children=[
    html.H2("Dashboard de Recomendaciones Musicales", style={'textAlign': 'center'}),
    
    # Contenedor de filtros: un slider para cada métrica
    html.Div(slider_controls, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
    
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
            data=[],
            page_size=10,
            style_table={'maxHeight': '300px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#222222', 'color': '#FFFFFF'},
            style_data={'backgroundColor': '#333333', 'color': '#FFFFFF'}
        )
    ], style={'padding': '10px'}),
    
    # Gráficos: uno de barras y uno de pastel
    html.Div([
        dcc.Graph(id='bar-chart'),
        dcc.Graph(id='pie-chart')
    ], style={'padding': '10px'})
])

# Callback: usar todos los sliders para filtrar el DataFrame
# Se crean inputs dinámicos para cada métrica
input_ids = [f"slider-{metric}" for metric in metrics]
@app.callback(
    [Output('rec-table', 'data'),
     Output('bar-chart', 'figure'),
     Output('pie-chart', 'figure')],
    [Input(id, 'value') for id in input_ids]
)
def update_dashboard(*slider_values):
    df = tracks_pd.copy()
    # Filtrar por cada métrica: para cada slider, se conserva solo filas con valor >= slider value
    for metric, val in zip(metrics, slider_values):
        df = df[df[metric] >= val]
    
    # Preparar datos para la tabla (mostrar hasta 50 registros)
    table_data = df[['name', 'artists', 'release_year', 'popularity', 'cluster']].head(50).to_dict('records')
    
    # Calcular métricas: en este ejemplo, solo diversidad y novedad (la precisión se omite)
    diversity_val = compute_diversity(df)
    novelty_val = 1 - (df['popularity'].mean()/100.0) if not df.empty else 0.0
    
    metrics_df = pd.DataFrame({
        'Métrica': ['Diversidad', 'Novedad'],
        'Valor': [diversity_val, novelty_val]
    })
    
    # Gráfico de barras
    bar_fig = px.bar(metrics_df, x='Métrica', y='Valor', color='Métrica',
                     title="Métricas del Motor de Recomendación",
                     color_discrete_sequence=['#1f77b4', '#2ca02c'])
    bar_fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    bar_fig.update_layout(template='plotly_dark', paper_bgcolor='#111111', plot_bgcolor='#111111',
                          yaxis_title="Valor", xaxis_title="Métrica", showlegend=False)
    
    # Gráfico de pastel
    pie_fig = px.pie(metrics_df, names='Métrica', values='Valor', 
                     title="Distribución de Métricas",
                     color='Métrica', color_discrete_sequence=['#1f77b4', '#2ca02c'])
    pie_fig.update_layout(template='plotly_dark', paper_bgcolor='#111111')
    
    return table_data, bar_fig, pie_fig

if __name__ == "__main__":
    print("Iniciando la aplicación en el puerto 8050...")
    time.sleep(3)
    print("La aplicación de recomendación se ha iniciado. Visita http://127.0.0.1:8050 en tu navegador.")
    app.run(host="0.0.0.0", port=8050, debug=True)