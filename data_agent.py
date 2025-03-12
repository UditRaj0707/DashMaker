from json import load
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import pandas as pd
from typing import Dict, List, Any
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from user_input import UserInput
import subprocess
import json

ADHOC_VIZ_PROMPT = PromptTemplate(
            input_variables=["data_info", "columns"],
            template="""Based on the following dataframe information:
            Columns: {columns}
            Data Preview: {data_info}
            
            Suggest 3 relevant visualizations that would best represent this data. Format your response as a list of visualization types with brief explanations and code using pandas and plotly keeping in mind the data types. 
            NOTE: Make sure to use the correct column names in the code."""
        )

ADHOC_DASHBOARD_PROMPT = PromptTemplate(
            input_variables=["visualizations"],
            template="""Given these visualization suggestions:
            {visualizations}
            
            Propose a dashboard layout that incorporates these visualizations effectively. Include:
            1. Layout structure
            2. Placement of each visualization
            3. Any interactive elements that would enhance user experience
            4. Color scheme suggestions
            Be specific but concise."""
        )

ADHOC_EDIT_VIZPROMPT = PromptTemplate(
            input_variables=["data_info", "columns", "old_visualizations", "user_query"],
            template="""Based on the following information:
            Columns: {columns}
            Data Preview: {data_info}

            Old Visualization Suggestions: {old_visualizations}

            And the NEW user query: {user_query}

            Your task is to make changes to the old visualization suggestions based on the new user query.
            
            Suggest relevant visualization changes according to the user query. If user asks for multiple changes, suggest one visualization for each change asked. Format your response as a list of visualization types with brief explanations and code using pandas and plotly. 
            Make sure to not remove old visualizations that user has not asked to remove.
            NOTE: Make sure to use the correct column names which are present in the data WHICH IS WHY THE DATA INFORMATION IS PROVIDED."""
        )

ADHOC_EDIT_DASHBOARD_PROMPT = PromptTemplate(
            input_variables=["old_dashboard_layout_recommendation", "user_query", "new_visualizations"],
            template="""Given the following inputs:
            1. Old Dashboard Layout Recommendation (This is the dashboard layout that was recommended before the user asked for changes): {old_dashboard_layout_recommendation}
            2. New User Query: {user_query}
            3. New Visualization Suggestions: {new_visualizations}
            
            Propose a dashboard layout that incorporates these visualizations effectively keeping in mind whether user has asked for replacing certain old visualizations with new ones or just adding new visualizations. If user has not spoken about replacing certain old visualizations, then do not remove those and incorporate them in the new dashboard layout.
            
            Include:
            1. Layout structure
            2. Placement of each visualization
            3. Any interactive elements that would enhance user experience
            4. Color scheme suggestions
            Be specific but concise.
            """
        )

ADHOC_DOC_VIZ_PROMPT = PromptTemplate(
            input_variables=["docs"],
            template="""You are a data analyst who suggests visualizations based on the following input:
            1. Documents containing financial tabular data: {docs}
            
            Suggest 3 relevant visualizations that would best represent this data. Format your response as a list of visualization types with brief explanations and code using pandas and plotly.   
            You will have to first analyze the data and think of visualizations and then write the code where insert the data required in a pandas dataframe and use plotly for visualization.
            
            Here are some examples to help you with:
            1. Input:
            | | Total | 2024 | 2025-2026 | 2027-2028 | 2029 and Thereafter |
            |---|---:|---:|---:|---:|---:|
            | Loans and notes payable:¹ | | | | | |
            | Commercial paper borrowings | $4,209 | $4,209 | $— | $— | $— |
            | Lines of credit and other short-term borrowings | 348 | 348 | — | — | — |
            | Current maturities of long-term debt² | 1,960 | 1,960 | — | — | — |
            | Long-term debt, net of current maturities² | 36,694 | — | 2,936 | 7,579 | 26,179 |
            | Estimated interest payments³ | 9,855 | 878 | 1,120 | 909 | 6,948 |
            | Accrued income taxes⁴ | 2,649 | 1,569 | 1,080 | — | — |
            | Purchase obligations⁵ | 23,392 | 13,701 | 3,330 | 2,057 | 4,304 |
            | Marketing obligations⁶ | 4,076 | 2,563 | 756 | 403 | 354 |
            | Lease obligations | 2,007 | 444 | 562 | 363 | 638 |
            | Acquisition obligations⁷ | 3,030 | 13 | 3,017 | — | — |
            | Held-for-sale and related obligations⁸ | 903 | 809 | 64 | 21 | 9 |
            | Total contractual obligations | $89,123 | $26,494 | $12,865 | $11,332 | $38,432 |

            Output:
            1. Stacked Bar Chart - Contractual Obligations Breakdown Over Time
                A stacked bar chart will show the proportion of different financial obligations over various time periods.
                This helps in understanding how financial commitments are distributed over the years.
            2. Line Chart - Long-term Debt and Interest Payments Trend
                A line chart comparing "Long-term debt, net of current maturities" and "Estimated interest payments" over different periods.
                Helps in understanding debt maturity patterns and related interest burden.
            3. Pie Chart - Share of Different Financial Obligations
                A pie chart representing the share of each category (e.g., long-term debt, purchase obligations, marketing obligations) in total contractual obligations.
                Useful for quickly assessing major cost drivers.

            Code: 
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go

            # Data Preparation
            data = {{
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
            }}

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

            Note: 
            1.There may be multiple tables in the input which you'd like to suggest visualizations, in that case maintain different dataframes in code.
            1. Do not overexplain the suggested visualization.
            3. Make sure the data in the code is same as the input. THIS IS VERY IMPORTANT.


            """
        )

ADHOC_DOC_DASHBOARD_PROMPT = PromptTemplate(
            input_variables=["visualizations"],
            template="""Given these visualization suggestions:
            {visualizations}
            
            Propose a dashboard layout that incorporates these visualizations effectively. Include:
            1. Layout structure
            2. Placement of each visualization
            3. Any interactive elements that would enhance user experience
            4. Color scheme suggestions
            Be specific but concise.
            """
        )

load_dotenv()

def run_code(code: str, file_path: str) -> str:
    script_filename = f"./scripts/data_analysis.py"
    full_script = f"""import pandas as pd
df = pd.read_csv('{file_path}')
{code}"""

    os.makedirs("./scripts", exist_ok=True)
    
    # Write the script
    with open(script_filename, "w", encoding='utf-8') as f:
        f.write(full_script)

    # Run the script and capture output
    try:
        result = subprocess.run(
            ["poetry", "run", "python", script_filename],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        return result.stdout.strip()
    except Exception as e:
        return f"Error executing script: {str(e)}"
    finally:
        # Clean up the temporary script
        if os.path.exists(script_filename):
            os.remove(script_filename)

class DataAnalyser:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.2
        )

        self.mode = None
        
        # Create chains using the new pipe syntax
        # self.visualization_chain = self.visualization_prompt | self.llm
        # self.dashboard_chain = self.dashboard_prompt | self.llm

    def init_chains(self, file_path: str):
        if file_path.endswith(".csv"):
            if self.mode == "adhoc-gen":
                self.visualization_prompt = ADHOC_VIZ_PROMPT
                self.dashboard_prompt = ADHOC_DASHBOARD_PROMPT
            elif self.mode == "adhoc-edit":
                self.visualization_prompt = ADHOC_EDIT_VIZPROMPT
                self.dashboard_prompt = ADHOC_EDIT_DASHBOARD_PROMPT
        elif file_path.endswith(".pdf"):
            if self.mode == "adhoc-gen":
                self.visualization_prompt = ADHOC_DOC_VIZ_PROMPT
                self.dashboard_prompt = ADHOC_DOC_DASHBOARD_PROMPT
            # elif self.mode == "adhoc-edit":
            #     self.visualization_prompt = ADHOC_EDIT_DOC_VIZPROMPT
            #     self.dashboard_prompt = ADHOC_EDIT_DOC_DASHBOARD_PROMPT

        self.visualization_chain = self.visualization_prompt | self.llm
        self.dashboard_chain = self.dashboard_prompt | self.llm

    def analyze_data(self, file_path: str, user_query: str = "", user_input: UserInput = None) -> Dict[str, Any]:
        """
        Analyze dataframe using provided run_code function and suggest visualizations
        """
        if self.mode == "adhoc-gen":
            if file_path.endswith(".csv"):
                # Get dataframe info
                info_command = "print(df.info())"
                df_info = run_code(info_command, file_path)
                
                # Get column names and types
                columns_command = "print(df.columns.tolist())"
                columns = run_code(columns_command, file_path)
                dtypes_command = "print(df.dtypes.to_string())"
                dtypes = run_code(dtypes_command, file_path)
                
                # Get data preview
                preview_command = "print(df.head().to_string())"
                preview = run_code(preview_command, file_path)
                
                # Get basic statistics
                stats_command = "print(df.describe(include='all').to_string())"
                stats = run_code(stats_command, file_path)
                
                # Analyze categorical columns
                cat_analysis_command = """
        cat_analysis = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            unique_vals = df[col].value_counts().head(10).to_dict()
            null_count = df[col].isnull().sum()
            cat_analysis[col] = {
                'unique_values': unique_vals,
                'total_unique': df[col].nunique(),
                'null_count': null_count
            }
        print(cat_analysis)
                """
                categorical_analysis = run_code(cat_analysis_command, file_path)
                
                # Analyze numerical columns
                num_analysis_command = """
        num_analysis = {}
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            stats = df[col].describe()
            null_count = df[col].isnull().sum()
            num_analysis[col] = {
                'min': stats['min'],
                'max': stats['max'],
                'mean': stats['mean'],
                'median': df[col].median(),
                'null_count': null_count,
                'skewness': df[col].skew()
            }
        print(num_analysis)
                """
                numerical_analysis = run_code(num_analysis_command, file_path)
                
                # Combine outputs
                code_output = f"""
                Data Info:
                {df_info}
                
                Column Types:
                {dtypes}
                
                Preview:
                {preview}
                
                Statistics:
                {stats}
                
                Categorical Columns Analysis:
                {categorical_analysis}
                
                Numerical Columns Analysis:
                {numerical_analysis}
                """

                self.data_info = code_output
                self.columns = columns
                
                # Get visualization suggestions
                viz_result = self.visualization_chain.invoke({
                    "data_info": self.data_info,
                    "columns": self.columns
                })
                self.viz_suggestions = viz_result.content if hasattr(viz_result, 'content') else str(viz_result)
                
                # Get dashboard suggestions
                dashboard_result = self.dashboard_chain.invoke({
                    "visualizations": self.viz_suggestions
                })
                self.dashboard_design = dashboard_result.content if hasattr(dashboard_result, 'content') else str(dashboard_result)
                
                return {
                    "data_summary": code_output,
                    "visualization_suggestions": self.viz_suggestions,
                    "dashboard_design": self.dashboard_design,
                    "detailed_analysis": {
                        "categorical": categorical_analysis,
                        "numerical": numerical_analysis
                    }
                }

            elif file_path.endswith(".pdf"):
                retrieved_docs = user_input.search("  ", k=2, filter=True)
                docs = "\n\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Get visualization suggestions
                viz_result = self.visualization_chain.invoke({
                    "docs": docs
                })
                self.viz_suggestions = viz_result.content if hasattr(viz_result, 'content') else str(viz_result)
                
                # Get dashboard suggestions
                dashboard_result = self.dashboard_chain.invoke({
                    "visualizations": self.viz_suggestions
                })
                self.dashboard_design = dashboard_result.content if hasattr(dashboard_result, 'content') else str(dashboard_result)

                output = {
                    "data_summary": docs,
                    "visualization_suggestions": self.viz_suggestions,
                    "dashboard_design": self.dashboard_design
                }

                with open(f"exp_analysis_doc_{self.mode}.jsonl", "a", encoding='utf-8') as f:
                    json_str = json.dumps(output)
                    f.write(json_str + '\n')
                
                return output

        elif self.mode == "adhoc-edit":
            if file_path.endswith(".csv"):
                print(f"User Query: {user_query}")
                print(f"OLD VISUALIZATIONS: {self.viz_suggestions}")
                print(f"Old Dashboard Design: {self.dashboard_design}")
                viz_result = self.visualization_chain.invoke({
                    "data_info": self.data_info,
                "columns": self.columns,
                "old_visualizations": self.viz_suggestions,
                "user_query": user_query
            })
            self.viz_suggestions = viz_result.content if hasattr(viz_result, 'content') else str(viz_result)
            
            # Get dashboard suggestions
            dashboard_result = self.dashboard_chain.invoke({
                "old_dashboard_layout_recommendation": self.dashboard_design,
                "user_query": user_query,
                "new_visualizations": self.viz_suggestions
            })
            self.dashboard_design = dashboard_result.content if hasattr(dashboard_result, 'content') else str(dashboard_result)
            
            return {
                "visualization_suggestions": self.viz_suggestions,
                "dashboard_design": self.dashboard_design
            }

    def invoke(self, file_path: str, mode="adhoc-gen", user_query: str = "", user_input: UserInput = None) -> Dict[str, Any]:
        # print(f"user query: {user_query}")
        self.mode = mode
        self.user_input = user_input

        self.init_chains(file_path)
        if file_path.endswith(".csv"):
            results = self.analyze_data(file_path, user_query)

            with open(f"exp_analysis_{mode}.json", "w", encoding='utf-8') as f:
                json.dump(results, f)
            return results
        
        # later for other file types
        elif file_path.endswith(".pdf"):
            results = self.analyze_data(file_path, user_query, user_input)
            return results

if __name__ == "__main__":
    data_analyser = DataAnalyser()
    result = data_analyser.invoke("data/dummy_data.csv")
    print(result)
