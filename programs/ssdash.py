

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import plotly.graph_objects as go

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    dcc.Graph(id='3dplot', 
              figure=go.Figure(data=[go.Surface(z=[[0, 1], [0, 1]])])),  # Replace with your own 3D data
    dcc.Dropdown(id='dropdown', 
                 options=[{'label': i, 'value': i} for i in ['id1', 'id2', 'id3']]),
    html.Button('SB', id='sb', n_clicks=0),
    html.Button('CPK', id='cpk', n_clicks=0),
    html.Button('BS', id='bs', n_clicks=0),
    html.Div(id='output')
])

# Define the app callbacks
@app.callback(
    Output('output', 'children'),
    [Input('sb', 'n_clicks'), Input('cpk', 'n_clicks'), Input('bs', 'n_clicks')]
)
def update_output(sb, cpk, bs):
    ctx = dash.callback_context

    if not ctx.triggered:
        return 'No button has been clicked yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        return f'You clicked {button_id}'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
