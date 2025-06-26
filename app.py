from dotenv import load_dotenv
import os
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from htmlTemplates import bot_template, user_template, css
import streamlit as st
from wordcloud import WordCloud
import re
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
# Load environment variables
load_dotenv()
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch
import warnings

def authenticate_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Authenticate locally
    drive = GoogleDrive(gauth)
    return drive

os.environ["OPENAI_API_KEY"] = 'API Key'
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'
os.environ["OPENAI_EMBEDDING_MODEL_NAME"] = 'text-embedding-3-large'

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")

def get_csv_text(csv_files):
    """Extract text content from CSV files."""
    text = ""
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            text += df.to_string(index=False)
        except Exception as e:
            st.error(f"Error reading file {csv_file.name}: {e}")
    return text

def get_chunk_text(text):
    """Split the text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create a vector store from text chunks."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    """Set up a conversational retrieval chain."""
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME, temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    system_template = """
    Use the following pieces of context and chat history to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Chat history: {chat_history}

    Question: {question}
    Helpful Answer:
    """
    prompt = PromptTemplate(
        template=system_template,
        input_variables=["context", "question", "chat_history"],
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        verbose=True,
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return conversation_chain

def handle_user_input(question):
    try:
        response = st.session_state.conversation({'question': question})
        st.session_state.chat_history = response['chat_history']
    except Exception as e:
        st.error('Please select CSV files and click on Process.')

def display_chat_history():
    if st.session_state.chat_history:
        reversed_history = st.session_state.chat_history[::-1]

        formatted_history = []
        for i in range(0, len(reversed_history), 2):
            chat_pair = {
                "AIMessage": reversed_history[i].content,
                "HumanMessage": reversed_history[i + 1].content
            }
            formatted_history.append(chat_pair)

        for i, message in enumerate(formatted_history):
            st.write(user_template.replace("{{MSG}}", message['HumanMessage']), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", message['AIMessage']), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title='Chat with CSVs', page_icon=':file_folder:')
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with CSVs :file_folder:')

    with st.sidebar:
        option = st.sidebar.selectbox(
            "Select an option:",
            ["Data Cleaning", "Classification Visualization", "LLM RAG Chatbot", "Upload CSV and Training",
             "Web Code Generation"]
        )
    if option == "LLM RAG Chatbot":
        question = st.text_input("Ask anything about your CSV data:")
        if question:
            handle_user_input(question)

        if st.session_state.chat_history is not None:
            display_chat_history()

        st.subheader("Upload your CSV Files Here: ")
        csv_files = st.file_uploader("Choose your CSV Files and Press Process button", type=['csv'], accept_multiple_files=True)

        if csv_files and st.button("Process"):
            with st.spinner("Processing your CSVs..."):
                try:
                    # Get CSV Text
                    raw_text = get_csv_text(csv_files)
                    # Get Text Chunks
                    text_chunks = get_chunk_text(raw_text)
                    # Create Vector Store
                    vector_store = get_vector_store(text_chunks)
                    st.success("Your CSVs have been processed. You can ask questions now.")
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    elif option == "Data Cleaning":
        st.subheader("Data Cleaning")
        st.title("Select the file that you want to clean")
        uploaded_file = st.file_uploader("Upload your uncleaned CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, engine='python', encoding='cp949')
            df.rename(columns={'결함내역': 'fault', '정비일자': 'repair_date', '작업목록': 'job', '정비항목': 'repair_detail',
                               '시작시간': 'start_time', '종료시간': 'end_time'}, inplace=True)
            df['time'] = pd.to_datetime(df['start_time'])
            df = df.set_index('time')
            df = df.sort_index()

            def find_english_words(text):
                # 정규 표현식을 사용하여 영단어 3개 이상 연속된 부분을 찾습니다.
                pattern = r"[a-zA-Z]{3,}"  # 영문자 3개 이상 연속된 패턴
                matches = re.findall(pattern, str(text))
                return " ".join(matches) if matches else np.nan

            df['english_words'] = df['repair_detail'].apply(find_english_words)  # 정비항목
            df_filtered = df.dropna(subset=['english_words'])
            df['english_words'].unique()
            # display the clean data
            st.subheader("Cleaned Data")
            st.dataframe(df_filtered)

            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Download Cleaned CSV",
                data=csv,
                file_name="cleaned.csv",
                mime="text/csv",
            )

    elif option == "Classification Visualization":
        st.subheader("Classification Visualization")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        st.write(css, unsafe_allow_html=True)
        if uploaded_file is not None:
            try:
                # Load the CSV file
                data = pd.read_csv(uploaded_file)

                # Check if required columns exist
                if "Keyword" in data.columns and "Frequency" in data.columns:
                    # Display the raw data
                    st.subheader("Uploaded Data")
                    st.dataframe(data)

                    # Group by keywords and sum their frequencies
                    keyword_data = data.groupby("Keyword", as_index=False)["Frequency"].sum()

                    # Display grouped data
                    st.subheader("Grouped Keyword Data")
                    st.dataframe(keyword_data)

                    # Prepare data for the pie chart
                    labels = keyword_data["Keyword"]
                    sizes = keyword_data["Frequency"]

                    # Generate the Word Cloud
                    st.subheader("Keyword Frequency Word Cloud")
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(
                        dict(zip(keyword_data["Keyword"], keyword_data["Frequency"]))
                    )
                    # Display the word cloud
                    st.image(wordcloud.to_array())

                else:
                    st.error("The CSV file must contain 'Keyword' and 'Frequency' columns.")
            except Exception as e:
                st.error(f"Error processing the file: {e}")
        else:
            st.info("Please upload a CSV file.")

    elif option == "Web Code Generation":
        st.subheader("Web Code Example")
        html_code = """
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                    }
                    .chat-container {
                        width: 50%;
                        margin: 20px auto;
                        border: 1px solid #ccc;
                        padding: 10px;
                        border-radius: 8px;
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                    }
                    .message {
                        margin: 5px 0;
                        padding: 10px;
                        border-radius: 5px;
                    }
                    .user-message {
                        background-color: #e6f7ff;
                        text-align: left;
                    }
                    .bot-message {
                        background-color: #f5f5f5;
                        text-align: left;
                    }
                </style>
            </head>
            <body>
                <div class="chat-container">
                    <div class="message user-message">
                        <b>User:</b> Hello, can you process this CSV file?
                    </div>
                    <div class="message bot-message">
                        <b>Bot:</b> Sure! Please upload the file and click "Process."
                    </div>
                </div>
            </body>
            </html>
            """
        # Display HTML code
        st.code(html_code, language="html")

    elif option == "Upload CSV and Training":
        warnings.filterwarnings('ignore')

        # Load the base Llama 2 model (use your actual model here)
        base_model = "beomi/llama-2-ko-7b"

        # Set the compute dtype for 4-bit quantization
        compute_dtype = getattr(torch, 'float16')

        # Define quantization configuration
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        # Load the model with quantization settings
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map={"": 0}  # Use GPU device 0
        )

        # Disable caching
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load the fine-tuned adapter
        model = PeftModel.from_pretrained(model, r"C:\Users\addmin\Downloads\ChatbotAPI\llama2_korean\code_fine_tuned")

        # Initialize the pipeline
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

        # Function to generate a response using the fine-tuned Llama 2 model with adapter
        def generate_response(prompt):
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            result = pipe(
                formatted_prompt,
                repetition_penalty=1.2,
                max_new_tokens=200,
                num_return_sequences=1
            )
            generated_text = result[0]['generated_text']
            cleaned_text = (
                generated_text.replace("<s>", "")
                .replace("[INST]", "")
                .replace("[/INST]", "")
                .strip()
            )
            return cleaned_text.replace(prompt, "").strip()

        # Streamlit App UI
        st.title("Chatbot with Llama 2")
        st.write("This chatbot is powered by a fine-tuned Llama 2 model.")

        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Input box for user query
        with st.form("chat_form"):
            user_input = st.text_input("Your message:", key="user_input", placeholder="Type your message here...")
            submit_button = st.form_submit_button("Send")

        # Generate response when the form is submitted
        if submit_button and user_input:
            st.session_state["messages"].append({"user": user_input})
            response = generate_response(user_input)
            st.session_state["messages"].append({"llama": response})

        # Display the chat history
        for msg in st.session_state["messages"]:
            if "user" in msg:
                st.write(f"**You:** {msg['user']}")
            elif "llama" in msg:
                col1, col2 = st.columns([1, 9])  # Adjust the column ratio as needed
                with col1:
                    st.image("petitrobot.png", width=50)  # Replace with your avatar image path
                with col2:
                    st.write(f"**Llama 2:** {msg['llama']}")
if __name__ == '__main__':
    main()