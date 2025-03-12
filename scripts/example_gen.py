import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load data
df = pd.read_csv("./data/Dashboard_data.csv")

# Data preprocessing
df['Revenue'] = df['Revenue'].str.replace(',', '').str.replace('$', '').astype(float)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y')

# Prepare data for plots
revenue_by_business_unit = df.groupby('Business Unit')['Revenue'].sum().reset_index()
ship_type_distribution = df['Ship Type'].value_counts().reset_index()
ship_type_distribution.columns = ['Ship Type', 'Count']
monthly_revenue = df.groupby(df['Ship Date'].dt.to_period('M'))['Revenue'].sum().reset_index()
monthly_revenue['Ship Date'] = monthly_revenue['Ship Date'].dt.to_timestamp()

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Business Performance Dashboard"),
    html.H2("Overview of Revenue and Shipments"),
    dcc.DatePickerRange(
        id="date-picker",
        start_date=df["Ship Date"].min(),
        end_date=df["Ship Date"].max()
    ),
    html.Div([
        html.Div([
            dcc.Graph(id="revenue-bar-chart")
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id="ship-type-pie-chart")
        ], style={'width': '48%', 'display': 'inline-block'}),
    ]),
    html.Div([
        dcc.Graph(id="monthly-revenue-line-chart")
    ], style={'width': '100%'}),
])

@app.callback(
    [dash.Output("revenue-bar-chart", "figure"),
     dash.Output("ship-type-pie-chart", "figure"),
     dash.Output("monthly-revenue-line-chart", "figure")],
    [dash.Input("date-picker", "start_date"),
     dash.Input("date-picker", "end_date")]
)
def update_charts(start_date, end_date):
    filtered_df = df[(df["Ship Date"] >= start_date) & (df["Ship Date"] <= end_date)]

    # Update data for plots based on date range
    revenue_by_business_unit = filtered_df.groupby('Business Unit')['Revenue'].sum().reset_index()
    ship_type_distribution = filtered_df['Ship Type'].value_counts().reset_index()
    ship_type_distribution.columns = ['Ship Type', 'Count']
    monthly_revenue = filtered_df.groupby(filtered_df['Ship Date'].dt.to_period('M'))['Revenue'].sum().reset_index()
    monthly_revenue['Ship Date'] = monthly_revenue['Ship Date'].dt.to_timestamp()

    # Bar Chart for Revenue by Business Unit
    fig1 = px.bar(revenue_by_business_unit, x='Business Unit', y='Revenue', title='Revenue by Business Unit',
                  color='Revenue', color_continuous_scale='Blues')

    # Pie Chart for Ship Type Distribution
    fig2 = px.pie(ship_type_distribution, names='Ship Type', values='Count', title='Ship Type Distribution',
                  color_discrete_sequence=['green', 'orange'])

    # Line Chart for Monthly Revenue Trend
    fig3 = px.line(monthly_revenue, x='Ship Date', y='Revenue', title='Monthly Revenue Trend', color_discrete_sequence=['teal'])

    return fig1, fig2, fig3

if __name__ == "__main__":
    app.run_server(debug=True, port=8000)