from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from load_dotenv import load_dotenv
load_dotenv()

ADHOC_PROMPT = """You are a coder with expertise in making dash apps for financial data visualization using plotly dash library.
You will be given these as inputs:
1. User Query: User's query for the dashboard (may or may not be given, empty string if not given)
2. Data Path: Path to the data file that contains the financial data.
3. Plot Recommendations: Plot recommendations are from a expert data anlayser who has recommended the best plots after thoroughly analyzing the data. 
4. Dashboard Layout Recommendations: Dashboard layout recommendations are from a expert financial dashboard designer who has recommended the best dash components after thoroughly analyzing the data and plot recommendations.

NOTE: You are required to use the same column names as given here. Do not use common column names, else later code will not work.

Your task is to generate the code for a dash app that visualizes the financial data based on the user query, the plot and dashboard layout recommendations.

Expected output:
A code that can be executed as-is and will generate a dash app that visualizes the financial data based on the user query, the plot and dashboard layout recommendations. Do not output anything else other than the code. Also use localhost 8000 port for the app that you will code.

Example 1:
User Query: "I want to analyze the stock price trends and compare volume traded across different companies."

Data Path: "./data/dummy_data.csv"

Plot Recommendations:
- A line chart to visualize stock price trends over time.
- A bar chart to compare the volume traded across different companies.

Dash Recommendations:
- A date range picker to filter stock data by time.
- A dropdown to select different companies.
- A graph component for the line chart.
- A graph component for the bar chart.

Output:
```python
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load data
df = pd.read_csv("./data/dummy_data.csv")

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.DatePickerRange(id="date-picker", start_date=df["Date"].min(), end_date=df["Date"].max()),
    dcc.Dropdown(id="company-dropdown", options=[{"label": c, "value": c} for c in df["Company"].unique()], multi=True),
    dcc.Graph(id="stock-trend"),
    dcc.Graph(id="volume-comparison"),
])

@app.callback(
    [dash.Output("stock-trend", "figure"), dash.Output("volume-comparison", "figure")],
    [dash.Input("date-picker", "start_date"), dash.Input("date-picker", "end_date"), dash.Input("company-dropdown", "value")]
)
def update_charts(start_date, end_date, selected_companies):
    filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    if selected_companies:
        filtered_df = filtered_df[filtered_df["Company"].isin(selected_companies)]

    fig1 = px.line(filtered_df, x="Date", y="Stock Price", color="Company", title="Stock Price Trend")
    fig2 = px.bar(filtered_df, x="Company", y="Volume", title="Volume Comparison")

    return fig1, fig2

if __name__ == "__main__":
    app.run_server(debug=True)
```

Example 2:
User Query: "I want to track daily percentage change in cryptocurrency prices."

Data Path: "./data/dummy_data.csv"

Plot Recommendations:
- A line chart to show daily percentage change of selected cryptocurrencies.
- A scatter plot to compare volatility across different cryptocurrencies.

Dash Recommendations:
- A dropdown to select cryptocurrencies.
- A slider to adjust the time window for analysis.
- A graph component for the line chart.
- A graph component for the scatter plot.

Output:
```python
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load data
df = pd.read_csv("./data/dummy_data.csv")

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(id="crypto-dropdown", options=[{"label": c, "value": c} for c in df["Crypto"].unique()], multi=True),
    dcc.Slider(id="time-window", min=5, max=30, step=5, value=10, marks={i: str(i) for i in range(5, 35, 5)}),
    dcc.Graph(id="pct-change-line"),
    dcc.Graph(id="volatility-scatter"),
])

@app.callback(
    [dash.Output("pct-change-line", "figure"), dash.Output("volatility-scatter", "figure")],
    [dash.Input("crypto-dropdown", "value"), dash.Input("time-window", "value")]
)
def update_charts(selected_cryptos, time_window):
    filtered_df = df.tail(time_window)
    if selected_cryptos:
        filtered_df = filtered_df[filtered_df["Crypto"].isin(selected_cryptos)]

    fig1 = px.line(filtered_df, x="Date", y="Pct Change", color="Crypto", title="Daily % Change in Crypto Prices")
    fig2 = px.scatter(filtered_df, x="Crypto", y="Volatility", title="Crypto Volatility Comparison")

    return fig1, fig2

if __name__ == "__main__":
    app.run_server(debug=True)
```


"""

ADHOC_EDITING_PROMPT = """You are a coder with expertise in making dash apps for financial data visualization using plotly dash library.
You will be given these as inputs:

1. Old Dash Code: The old dash code that was generated for the original user query.
2. New User Query: The new user query that you need to use to generate the dash app.
3. Data Path: Path to the data file that contains the financial data.
4. New Plot Recommendations: The new plot recommendations that you need to use to generate the dash app.
5. New Dashboard Layout Recommendations: The new dashboard layout recommendations that you need to use to generate the dash app.

You need to generate the new dash code for the new user query according to the new plot and dashboard layout recommendations. Make sure the changes you make to the old code to get the new code are minimal and only based on the new user query and do not make any changes that the user has not asked for.

Expected output:
A code that can be executed as-is and will generate a dash app that visualizes the financial data based on the new user query, the new plot and dashboard layout recommendations. Do not output anything else other than the code. Also use localhost 8000 port for the app that you will code.
"""

ADHOC_DOC_PROMPT = """You are a coder with expertise in making dash apps for financial data visualization using plotly dash library.
You will be given these as inputs:
1. User Query: User's query for the dashboard (may or may not be given, empty string if not given)
2. Plot Recommendations: Plot recommendations are from a expert data anlayser who has recommended the best plots after thoroughly analyzing the data and along with it the code for visualization. 
3. Dashboard Layout Recommendations: Dashboard layout recommendations are from a expert financial dashboard designer who has recommended the best dash components after thoroughly analyzing the data and plot recommendations.

Your task is to generate the code for a dash app that visualizes the financial data based on the user query, the plot and dashboard layout recommendations.

NOTE: Use the same code as provided with visualization suggestions, and most importantly do not alter the data.

Expected output:
A code that can be executed as-is and will generate a dash app that visualizes the financial data based on the user query, the plot and dashboard layout recommendations. Do not output anything else other than the code. Also use localhost 8000 port for the app that you will code.

Example 1:
User Query: "I want to analyze financial obligations across different time periods."

Plot Recommendations:
- A stacked bar chart to visualize the breakdown of contractual obligations over time.
- A line chart to track long-term debt and estimated interest payments over time.
- A pie chart to represent the proportion of different financial obligations.

Dash Recommendations:
- A dropdown to select different financial obligation categories.
- A range slider to filter data by time periods.
- A graph component for the stacked bar chart.
- A graph component for the line chart.
- A graph component for the pie chart.

Code:
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Data Preparation
data = {
    "Category": [
        "Commercial paper borrowings", "Lines of credit", "Current maturities of long-term debt",
        "Long-term debt, net of current maturities", "Estimated interest payments",
        "Accrued income taxes", "Purchase obligations", "Marketing obligations",
        "Lease obligations", "Acquisition obligations", "Held-for-sale obligations"
    ],
    "Total": [4209, 348, 1960, 36694, 9855, 2649, 23392, 4076, 2007, 3030, 903],
    "2024": [4209, 348, 1960, 0, 878, 1569, 13701, 2563, 444, 13, 809],
    "2025-2026": [0, 0, 0, 2936, 1120, 1080, 3330, 756, 562, 3017, 64],
    "2027-2028": [0, 0, 0, 7579, 909, 0, 2057, 403, 363, 0, 21],
    "2029 and Thereafter": [0, 0, 0, 26179, 6948, 0, 4304, 354, 638, 0, 9]
}

df = pd.DataFrame(data)

# **Stacked Bar Chart**
fig1 = px.bar(df, x="Category", y=["2024", "2025-2026", "2027-2028", "2029 and Thereafter"],
              title="Contractual Obligations Breakdown Over Time", barmode="stack")
fig1.show()

# **Line Chart**
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=["2024", "2025-2026", "2027-2028", "2029 and Thereafter"], 
                          y=[0, 2936, 7579, 26179], mode='lines+markers', name="Long-term Debt"))
fig2.add_trace(go.Scatter(x=["2024", "2025-2026", "2027-2028", "2029 and Thereafter"], 
                          y=[878, 1120, 909, 6948], mode='lines+markers', name="Interest Payments"))
fig2.update_layout(title="Long-term Debt and Interest Payments Trend")
fig2.show()

# **Pie Chart**
fig3 = px.pie(df, values="Total", names="Category", title="Share of Different Financial Obligations")
fig3.show()

Output:
```python
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Sample Data
data = {
    "Category": [
        "Commercial paper borrowings", "Lines of credit", "Current maturities of long-term debt",
        "Long-term debt, net of current maturities", "Estimated interest payments",
        "Accrued income taxes", "Purchase obligations", "Marketing obligations",
        "Lease obligations", "Acquisition obligations", "Held-for-sale obligations"
    ],
    "Total": [4209, 348, 1960, 36694, 9855, 2649, 23392, 4076, 2007, 3030, 903],
    "2024": [4209, 348, 1960, 0, 878, 1569, 13701, 2563, 444, 13, 809],
    "2025-2026": [0, 0, 0, 2936, 1120, 1080, 3330, 756, 562, 3017, 64],
    "2027-2028": [0, 0, 0, 7579, 909, 0, 2057, 403, 363, 0, 21],
    "2029 and Thereafter": [0, 0, 0, 26179, 6948, 0, 4304, 354, 638, 0, 9]
}
df = pd.DataFrame(data)

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(id="category-dropdown", options=[{"label": c, "value": c} for c in df["Category"].unique()], multi=True),
    dcc.RangeSlider(id="time-slider", min=0, max=4, step=1, marks={0: "2024", 1: "2025-2026", 2: "2027-2028", 3: "2029 and Thereafter"}, value=[0, 3]),
    dcc.Graph(id="stacked-bar"),
    dcc.Graph(id="line-chart"),
    dcc.Graph(id="pie-chart"),
])

@app.callback(
    [dash.Output("stacked-bar", "figure"), dash.Output("line-chart", "figure"), dash.Output("pie-chart", "figure")],
    [dash.Input("category-dropdown", "value"), dash.Input("time-slider", "value")]
)
def update_charts(selected_categories, time_range):
    period_columns = ["2024", "2025-2026", "2027-2028", "2029 and Thereafter"]
    selected_periods = period_columns[time_range[0]:time_range[1] + 1]
    
    filtered_df = df if not selected_categories else df[df["Category"].isin(selected_categories)]
    
    # Stacked Bar Chart
    fig1 = px.bar(filtered_df, x="Category", y=selected_periods, title="Breakdown of Financial Obligations Over Time", barmode="stack")
    
    # Line Chart for Long-term Debt and Interest Payments
    fig2 = px.line(df[df["Category"].isin(["Long-term debt, net of current maturities", "Estimated interest payments"])], 
                   x=period_columns, y="Total", color="Category", title="Long-term Debt vs Interest Payments")
    
    # Pie Chart for Obligation Share
    fig3 = px.pie(filtered_df, values="Total", names="Category", title="Proportion of Financial Obligations")
    
    return fig1, fig2, fig3

if __name__ == "__main__":
    app.run_server(debug=True, port=8000)
```

"""

class DashCoder:
    def __init__(self):
        self.mode = None
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3,
        )

    def init_prompts(self, mode, file_path):
        if file_path.endswith(".csv"):
            if mode == "adhoc-gen":
                self.system_prompt = ADHOC_PROMPT
            elif mode == "adhoc-edit":
                self.system_prompt = ADHOC_EDITING_PROMPT
        elif file_path.endswith(".pdf"):
            if mode == "adhoc-gen":
                self.system_prompt = ADHOC_DOC_PROMPT
            # elif mode == "adhoc-edit":
            #     self.system_prompt = ADHOC_EDIT_DOC_PROMPT

    def invoke(self, query, data_path, plot_recommendations, dash_recommendations, mode, old_code=""): 
        self.init_prompts(mode, data_path)
        print(mode)
        if data_path.endswith(".csv"):
            user_prompt = f"User: {query}\n\n" + \
                "Data Path:\n" + \
                data_path + \
                "\n\n" + \
                "Plot Recommendations:\n" + \
                plot_recommendations + \
                "\n\n" + \
                "Dash Recommendations:\n" + \
                dash_recommendations
            if mode == "adhoc-edit":
                user_prompt += f"\n\nOld Dash Code:\n{old_code}"
            
            messages = [
                SystemMessage(self.system_prompt),
                HumanMessage(user_prompt)
            ]

            result = self.llm.invoke(messages)
            code = result.content.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()
            return code
        elif data_path.endswith(".pdf"):
            user_prompt = f"User: {query}\n\n" + \
                "Plot Recommendations:\n" + \
                plot_recommendations + \
                "\n\n" + \
                "Dash Recommendations:\n" + \
                dash_recommendations
            if mode == "adhoc-edit":
                user_prompt += f"\n\nOld Dash Code:\n{old_code}"
            
            messages = [
                SystemMessage(self.system_prompt),
                HumanMessage(user_prompt)
            ]

            result = self.llm.invoke(messages)
            code = result.content.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()
            return code
    

    

# write a main guard
if __name__ == "__main__":
    agent = DashCoder()
    exp_query = """Generate a financial reporting dashboard using the Plotly Dash library. The dashboard should contain the following components:

Title Section:

Display "Financial Reporting" at the top.
Use a navigation bar with a placeholder for a company logo (e.g., "FinAI" in the top right corner).
Revenue Analysis:

Pie Chart: Visualize the revenue distribution by division, including categories like "Hedge Fund Strategies," "Capital Solutions," and "Customized Credit Strategies." Ensure each segment is labeled with percentage values.
Bar and Line Chart: Display a combination of revenue and operating margin over months.
Use a bar chart for monthly revenues.
Overlay a line chart for operating margin percentage on a secondary y-axis.
Expense Table:

Display a data table summarizing different expense types (e.g., "Business Development," "Depreciation and Amortization," "General Administrative and Other").
Include monthly totals for each category.
Enable a dropdown filter to select expense types and a month-wise filter.
Error Handling & Messages:

Show an error message placeholder for missing data (e.g., "Invalid visualization: The visualization was not found on the server").
Interactivity:

Use a dropdown menu to filter the expense table by expense type.
Allow users to select a specific month from a dropdown to update the revenue vs. operating margin chart dynamically.
Styling and Layout:

Use a two-column layout: One for revenue breakdown (pie chart) and one for revenue vs. margin analysis (bar + line chart).
Place the expenses table below with a scrollable feature.
Ensure all figures are formatted in currency notation.
Use Plotly Dash and Dash DataTable for interactivity. Ensure that the layout is responsive and well-structured."""
    response = agent.invoke(exp_query)
    print(response)
    agent.execute_code(response)