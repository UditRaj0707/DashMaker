import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Data Preparation
data = {
    "Year": ["2023", "2022", "2021"],
    "Net Operating Revenues": [45754, 43004, 38655],
    "Gross Profit": [27234, 25004, 23298],
    "Operating Income": [11311, 10909, 10308],
    "Net Income Attributable to Shareowners": [10714, 9542, 9771],
    "Cost of Goods Sold": [18520, 18000, 15357],
    "Selling, General and Administrative Expenses": [13972, 12880, 12144],
    "Other Operating Charges": [1951, 1215, 846]
}

df = pd.DataFrame(data)

# Line Chart - Net Operating Revenues and Gross Profit Trend
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df["Year"], y=df["Net Operating Revenues"], mode='lines+markers', name="Net Operating Revenues", line=dict(color='#F40009')))
fig1.add_trace(go.Scatter(x=df["Year"], y=df["Gross Profit"], mode='lines+markers', name="Gross Profit", line=dict(color='#006400')))
fig1.update_layout(title="Net Operating Revenues and Gross Profit Trend")

# Bar Chart - Operating Income and Net Income Comparison
fig2 = go.Figure(data=[
    go.Bar(name='Operating Income', x=df["Year"], y=df["Operating Income"], marker_color='#1E90FF'),
    go.Bar(name='Net Income Attributable to Shareowners', x=df["Year"], y=df["Net Income Attributable to Shareowners", marker_color='#FFA500'])
])
fig2.update_layout(barmode='group', title="Operating Income and Net Income Comparison")

# Pie Chart - Expense Distribution for 2023
expenses_2023 = {
    "Category": ["Cost of Goods Sold", "Selling, General and Administrative Expenses", "Other Operating Charges"],
    "Amount": [18520, 13972, 1951]
}
df_expenses_2023 = pd.DataFrame(expenses_2023)

fig3 = px.pie(df_expenses_2023, values="Amount", names="Category", title="Expense Distribution for 2023",
              color_discrete_map={"Cost of Goods Sold": '#FF6347', 
                                  "Selling, General and Administrative Expenses": '#32CD32', 
                                  "Other Operating Charges": '#9370DB'})

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#F5F5F5', 'padding': '10px'}, children=[
    html.H1("The Coca-Cola Company Financial Dashboard", style={'textAlign': 'center', 'color': '#333333'}),
    html.H3("Financial Performance and Expense Analysis (2021-2023)", style={'textAlign': 'center', 'color': '#333333'}),
    
    html.Div([
        html.Div([
            dcc.Graph(figure=fig1),
            html.P("This chart shows the trend of Net Operating Revenues and Gross Profit over the years, highlighting revenue growth and profitability trends.", style={'color': '#333333'})
        ], style={'width': '60%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(figure=fig2),
            html.P("This chart compares Operating Income and Net Income Attributable to Shareowners, providing insights into operating performance and net profitability.", style={'color': '#333333'})
        ], style={'width': '40%', 'display': 'inline-block', 'padding': '10px'})
    ]),
    
    html.Div([
        dcc.Graph(figure=fig3),
        html.P("This pie chart represents the distribution of major expenses for 2023, helping to assess the major cost components for the year.", style={'color': '#333333'})
    ], style={'width': '100%', 'padding': '10px'})
])

if __name__ == "__main__":
    app.run_server(debug=True, port=8000)