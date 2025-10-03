import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px

# Load and preprocess the dataset
data = pd.read_csv("./Data/final.csv")
data['time'] = pd.to_datetime(data['time'])  # Parse the time column as datetime

# Create Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div(
    children=[
    html.H1(children='Energy Dashboard', style={'textAlign': 'center', 'margin-bottom': '20px'}),

    # Dropdown to select a metric
    html.Label("Select a metric to visualize:"),
    dcc.Dropdown(
        id="metric-select",
        options=[{"label": column, "value": column} for column in data.columns if column != 'time'],
        value='generation biomass'  # Default metric
    ),

    # Line chart to display the metric
    dcc.Graph(id='metric-graph'),

    # Add spacing between the graph and date range picker
    html.Div(style={'height': '30px'}),  # Spacer

    # Date range picker to filter data
    html.Label("Select a date range:"),
    dcc.DatePickerRange(
        id='date-range-picker',
        start_date=data['time'].min(),
        end_date=data['time'].max(),
        display_format="YYYY-MM-DD"
    ),

    # Button to reset the date range
    html.Div(style={'margin-top': '10px'}),  # Add small spacing above the button
    html.Button('Reset Date Range', id='reset-button'),
])

# Callback for updating the line chart
@app.callback(
    dash.dependencies.Output('metric-graph', 'figure'),
    dash.dependencies.Input('metric-select', 'value'),
    dash.dependencies.Input('date-range-picker', 'start_date'),
    dash.dependencies.Input('date-range-picker', 'end_date'))
def update_graph(selected_metric, start_date, end_date):
    filtered_data = data[(data['time'] >= start_date) & (data['time'] <= end_date)]
    figure = px.line(filtered_data, x="time", y=selected_metric, title=f"{selected_metric} Over Time")
    return figure

# Callback for resetting the date range
@app.callback(
    [dash.dependencies.Output('date-range-picker', 'start_date'),
     dash.dependencies.Output('date-range-picker', 'end_date')],
    dash.dependencies.Input('reset-button', 'n_clicks'))
def reset_date_range(n_clicks):
    return data['time'].min(), data['time'].max()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
