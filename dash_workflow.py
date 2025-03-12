from langgraph.graph import StateGraph
from langgraph.graph import END
from typing import Annotated, Dict, TypedDict
from pydantic import BaseModel
from data_agent import DataAnalyser
from dash_agent import DashCoder
from user_input import UserInput
import os
import psutil
import subprocess
import sys

class DashFlowState(BaseModel):
    mode: str = ""
    query: str =   ""
    file_path: str = ""
    visualization_suggestions: str = ""
    dashboard_design: str = ""
    dash_code: str = ""
    url: str = ""


def kill_existing_process(port=8000):
    """Kill any process running on the given port."""
    for proc in psutil.process_iter(['pid', 'name']):  
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    proc.terminate()
                    print(f"Killed process {proc.name()} (PID: {proc.pid}) running on port {port}")
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass  

def execute_code(generated_code):
    """Write and run the generated Dash code in a new terminal."""
    script_filename = "./scripts/generated_dash_app.py"
    os.makedirs("./scripts", exist_ok=True)
    
    with open(script_filename, "w", encoding='utf-8') as f:
        f.write(generated_code)

    kill_existing_process(8000)  

    if sys.platform == "win32":
        cmd = f'start cmd /k "poetry run python {script_filename}"'
    elif sys.platform == "darwin": 
        cmd = f'osascript -e \'tell application "Terminal" to do script "poetry run python {script_filename}"\''
    else: 
        cmd = f'gnome-terminal -- bash -c "poetry run python {script_filename}; exec bash"'

    subprocess.Popen(cmd, shell=True)  

    return "http://localhost:8000"

    

class DashWorkflow:
    def __init__(self):
        self.data_analyser = DataAnalyser()
        self.dash_maker = DashCoder()
        self.user_input = UserInput()
        self.workflow = self.create_graph()
        self.app = self.workflow.compile()

    def create_graph(self):
        self.workflow =  StateGraph(DashFlowState)
        self.workflow.add_node("file_processing", self.file_processing)
        self.workflow.add_node("data_exploration", self.data_exploration)
        self.workflow.add_node("dash_app_generation", self.dash_app_generation)
        self.workflow.add_node("deployment", self.deployment)
        self.workflow.add_edge("file_processing", "data_exploration")
        self.workflow.add_edge("data_exploration", "dash_app_generation")
        self.workflow.add_edge("dash_app_generation", "deployment")
        self.workflow.add_edge("deployment", END)
        self.workflow.set_entry_point("file_processing")
        return self.workflow

    def file_processing(self, state):
        print("Processing file...")
        if state.file_path.endswith(".pdf") and state.mode == "adhoc-gen":
            self.user_input.process_files(state.file_path)
        return state

    def data_exploration(self, state):
        print("Exploring data...")
        # print(f"state.query: {state.query}")
        if state.file_path.endswith(".pdf") and state.mode == "adhoc-gen":
            results = self.data_analyser.invoke(state.file_path, mode=state.mode, user_query=state.query, user_input=self.user_input)
        elif state.file_path.endswith("csv"):
            results = self.data_analyser.invoke(state.file_path, mode=state.mode, user_query=state.query)
        state.visualization_suggestions = results["visualization_suggestions"]
        state.dashboard_design = results["dashboard_design"]
        return state

    def dash_app_generation(self, state):
        print("Generating dash app...")
        if state.mode == "adhoc-gen":
            code = self.dash_maker.invoke(state.query, state.file_path, state.visualization_suggestions, state.dashboard_design, mode=state.mode)
        elif state.mode == "adhoc-edit":
            code = self.dash_maker.invoke(state.query, state.file_path, state.visualization_suggestions, state.dashboard_design, mode=state.mode, old_code=state.dash_code)
        state.dash_code = code
        return state

    def deployment(self, state):
        print("Deploying dash app...")
        url = execute_code(state.dash_code)
        state.url = url
        return state

    def run(self, query, file_path, mode="adhoc-gen"):
        config = {"query": query, "file_path": file_path, "mode": mode}
        return self.app.invoke(config)

def main():
    workflow = DashWorkflow()
    info_for_user = """
    1. Generate a dashboard
    2. Edit a dashboard
    3. Exit
    """
    
    while True:
        print(info_for_user)
        mode = int(input("Enter what you want to do: "))
        
        if mode == 1:
            mode = "adhoc-gen"
            query = input("Enter query: ")
            if not query:
                query = "Generate a financial dashboard"
            file_path = input("Enter file path: ")
            if not file_path:
                file_path = "data/form-10k-exp.pdf"
            result = workflow.run(query, file_path, mode)
            print(f"Dashboard deployed at {result['url']}")
            
        elif mode == 2:
            mode = "adhoc-edit"
            query = input("Enter query: ")
            if not query:
                query = "Generate a financial dashboard"
            file_path = input("Enter file path: ")
            if not file_path:
                file_path = "data/form-10k-exp.pdf"
            result = workflow.run(query, file_path, mode)
            print(f"Dashboard deployed at {result['url']}")
        elif mode == 3:
            break
        else:
            print("Invalid mode")

if __name__ == "__main__":
    output = main()