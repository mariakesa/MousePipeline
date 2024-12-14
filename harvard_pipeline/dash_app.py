import pickle
import base64
import os
from io import BytesIO

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.manifold import TSNE
import numpy as np

# Load data
dat = pickle.load(open('/home/maria/Documents/HarvardData/processed_sessions_v3/Bo220226/session_responses.p','rb'))
session_ims = pickle.load(open('/home/maria/Documents/HarvardData/processed_sessions_v3/Bo220226/session_images.p','rb'))

assert dat.shape[0] == len(session_ims)

# Run TSNE on the neural responses
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedding = tsne.fit_transform(dat)  # shape (1250, 2)

# Convert image paths by removing 'OOD_monkey_data/Images/' prefix
base_image_dir = '/home/maria/Documents/HarvardData/Images'
image_paths = []
for p in session_ims:
    # Extract the filename after 'OOD_monkey_data/Images/'
    filename = p.split('OOD_monkey_data/Images/')[-1]
    # Split the extension off and re-append '.jpg' in lowercase
    base_name, ext = os.path.splitext(filename)
    filename = base_name + '.jpg'  # force lowercase .jpg extension
    full_path = os.path.join(base_image_dir, filename)
    image_paths.append(full_path)

# Create Plotly figure for TSNE embedding
fig = px.scatter(
    x=embedding[:,0],
    y=embedding[:,1],
    title='t-SNE Embedding of Neural Responses'
)

# Attach image_paths to customdata so we can easily retrieve the path on hover
fig.update_traces(customdata=image_paths)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("t-SNE Embedding of Neural Responses"),
    html.Div([
        html.Div([
            dcc.Graph(
                id='tsne-graph',
                figure=fig,
                style={'height': '80vh'}
            )
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.H3("Hovered Image"),
            html.Div(id='image-display')
        ], style={'width': '25%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'})
    ])
])

@app.callback(
    Output('image-display', 'children'),
    Input('tsne-graph', 'hoverData')
)
def display_hovered_image(hoverData):
    if hoverData is None:
        return html.Div("Hover over a point to see the corresponding image.")

    # hoverData format: {'points': [{'curveNumber':0, 'pointIndex': idx, 'pointNumber': idx, 'customdata': <image_path>}, ...]}
    # We stored the image path in customdata.
    point_data = hoverData['points'][0]
    img_path = point_data.get('customdata', None)

    if img_path is None:
        return html.Div("No image path found in hover data.")

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return html.Div("Image not found on disk.")

    # Encode image in base64
    with open(img_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('ascii')

    return html.Img(src='data:image/jpg;base64,{}'.format(encoded_image),
                    style={'maxWidth': '100%'})

if __name__ == '__main__':
    app.run_server(debug=True)
