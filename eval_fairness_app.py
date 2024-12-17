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

st.set_page_config(page_title="Document Genie", layout="wide")

def csv_uploader(uploaded_file):
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        return dataframe
def get_inspeq_evaluation(prompt, response, context, metric):
    inspeq_eval = InspeqEval(inspeq_api_key=st.session_state['INSPEQ_API_KEY'], inspeq_project_id= st.session_state['INSPEQ_PROJECT_ID'])
    input_data = [{
    "prompt": prompt,
    "response": response,
    "context": context
        }]
    metrics_list = metric
    try:
        output = inspeq_eval.evaluate_llm_task(
            metrics_list=metrics_list,
            input_data=input_data,
            task_name="task"
        )
        return output
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def build_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = st.session_state['chunk_size'] , chunk_overlap= st.session_state['chunk_overlap'])
    text_chunks = text_splitter.split_text(text)
    st.session_state['vector_store']= LanceDB.from_texts(text_chunks, st.session_state["embed_model"])

def fetch_context(query):
    return st.session_state['vector_store'].similarity_search(query, k = st.session_state['top_k'])

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "I don't think the answer is available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=st.session_state['api_key'])
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def llm_output(chain, docs, user_question):
    return chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)


def user_input(user_question):
    contexts_with_scores, exec_time = fetch_context(user_question)
    st.session_state["eval_models"]["app_metrics"].exec_times["chunk_fetch_time"] = exec_time

    chain = get_conversational_chain()
    response, exec_time = llm_output(chain, contexts_with_scores, user_question)
    st.session_state["eval_models"]["app_metrics"].exec_times["llm_resp_time"] = exec_time
    
    st.write("Reply: ", response["output_text"])

    ctx = ""
    for item in contexts_with_scores:
        if len(item.page_content.strip()):
            ctx += f"<br>{item.page_content}<br>&nbsp</li>"

    with st.expander("Click to see the context passed"):
        st.markdown(f"""<ol>{ctx}</ol>""", unsafe_allow_html=True)
    
    return contexts_with_scores, response["output_text"]

def result_to_list(results):
    eval = []
    score = []
    label = []

    return eval, score, label
def evaluate_all(query, context_lis, response, metrics_list):

    context = "\n\n".join(context_lis) if len(context_lis) else "no context"
    
    RESULT = {}

    RESULT["guards"] = {
        "evaluations" : get_inspeq_evaluation(query, response, context, metrics_list)
    }
    RESULT["execution_times"] = (st.session_state["eval_models"]["app_metrics"].exec_times)
    
    return RESULT


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
                    "Select Privileged Groups", 
                    st.session_state["data"]["Attribute"].unique().tolist(),
                    default=[]
                )
                st.session_state['privileged'] = st.multiselect(
                    "Select Privileged Groups",
                    st.session_state["data"]["Group"].unique().tolist(),
                    default=[]
                )
                st.session_state['un_privileged'] = st.multiselect(
                    "Select Unprivileged Groups",
                    st.session_state["data"]["Group"].unique().tolist(),
                    default=[]
                )
            submit_button = st.form_submit_button(label="Evaluate Fairness")
        
    if submit_button:
        st.session_state['filtered_data'] = input_parser(
        threshold=float(st.session_state["threshold"]), 
        data = st.session_state["data"], 
        metric_name = st.session_state["metric"][0], 
        attribute= st.session_state["attribute"], 
        previledged=st.session_state["privileged"], 
        unpreviledged=st.session_state["un_privileged"])
        st.session_state["fairness_score"] = fairness_score(st.session_state['filtered_data'], st.session_state["attribute"])
    if st.session_state["excel"]:
        st.write(st.session_state["data"])
    if st.session_state["fairness_score"]:
        fairness_df, final_fairness = st.session_state["fairness_score"]
        st.write(fairness_df)
        st.write(f"Fairness Score is \n {final_fairness}")
if __name__ == "__main__":
    main()
