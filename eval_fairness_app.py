import streamlit as st
import pandas as pd
from inspeq.client import InspeqEval
from io import StringIO
from fairness_score import fairness_score, input_parser

if 'api_key' not in st.session_state: st.session_state['api_key'] = None
if 'INSPEQ_API_KEY' not in st.session_state: st.session_state['INSPEQ_API_KEY'] = None
if 'INSPEQ_PROJECT_ID' not in st.session_state: st.session_state['INSPEQ_PROJECT_ID'] = None
if 'user_turn' not in st.session_state: st.session_state['user_turn'] = False
if 'pdf' not in st.session_state: st.session_state['pdf'] = None
if "embed_model" not in st.session_state: st.session_state['embed_model'] = None
if "vector_store" not in st.session_state: st.session_state['vector_store'] = None
if "metrics" not in st.session_state: st.session_state['metrics'] = None
if "options" not in st.session_state: st.session_state['options'] = []
if "excel" not in st.session_state: st.session_state['excel'] = None
if "metric" not in st.session_state: st.session_state['metric'] = None
if "threshold" not in st.session_state: st.session_state['threshold'] = None
if "data" not in st.session_state: st.session_state['data'] = None
if "attribute" not in st.session_state: st.session_state['attribute'] = None
if "privileged" not in st.session_state: st.session_state['privileged'] = None
if "un_privileged" not in st.session_state: st.session_state['un_privileged'] = None
if "filtered_data" not in st.session_state: st.session_state['filtered_data'] = None
if "fairness_score" not in st.session_state: st.session_state['fairness_score'] = None
if "percentage" not in st.session_state: st.session_state['percentage'] = None

st.set_page_config(page_title="Document Genie", layout="wide")

def csv_uploader(uploaded_file):
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        return dataframe

def main():
    st.markdown("""## Inspeq Fairness Demo""")

    with st.sidebar:
        st.title("Menu:")
        if "options" not in st.session_state:  # Ensure session state is initialized
            st.session_state["options"] = []
        st.session_state["excel"] = st.file_uploader("Choose a file",  key="csv_uploader")
        st.session_state["data"] = csv_uploader(st.session_state["excel"])
        with st.form(key="fairness_form"):
            if st.session_state["data"] is not None:
                # Initialize multiselect with valid data
                st.session_state['metric'] = st.multiselect(
                    "Select Metrics to Evaluate",
                    st.session_state["data"]["Metric"].unique().tolist(),default = []
                )
                st.session_state['threshold'] = st.text_input(
                    "Enter Threshold Value", key="threshold_value"
                )
                st.session_state['attribute'] = st.multiselect(
                    "Select Attribute", 
                    st.session_state["data"]["Attribute"].unique().tolist(),
                    default=[]
                )
                st.session_state['privileged'] = st.multiselect(
                    "Select Protected Group 1",
                    st.session_state["data"]["Group"].unique().tolist(),
                    default=[]
                )
                st.session_state['un_privileged'] = st.multiselect(
                    "Select Protected Group 2",
                    st.session_state["data"]["Group"].unique().tolist(),
                    default=[]
                )
                st.session_state['percentage'] = st.text_input(
                    "Select percentage of data to be selected:",
                    key = "percentage_value"
                )
            submit_button = st.form_submit_button(label="Evaluate Fairness")
        
    if submit_button:
        st.session_state['filtered_data'] = input_parser(
        threshold=float(st.session_state["threshold"]), 
        data = st.session_state["data"], 
        metric_name = st.session_state["metric"][0], 
        attribute= st.session_state["attribute"], 
        previledged=st.session_state["privileged"], 
        unpreviledged=st.session_state["un_privileged"],
        percentage = float(st.session_state['percentage'])
        ),
        st.session_state["fairness_score"] = fairness_score(st.session_state['filtered_data'][0], st.session_state["attribute"])
        if st.session_state["excel"]:
            st.write(st.session_state["filtered_data"][0])
        if st.session_state["fairness_score"]:
            fairness_df, final_fairness = st.session_state["fairness_score"]
            st.write(fairness_df)
            st.write(f"Fairness Score is \n {final_fairness}")
if __name__ == "__main__":
    main()
