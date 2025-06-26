# CSV Chat using RAG

### Finding proper answer from CSV content.

This project aims to develop a chatbot capable of interacting with users and providing precise answers from a csv file. By leveraging natural language processing and machine learning techniques, the chatbot can comprehend user queries and retrieve relevant information efficiently. Utilizing OpenAI models, the chatbot harnesses advanced language models and embeddings to enhance conversational capabilities and deliver accurate responses.

## Features

- **Support for cv**: Users can upload and query information from csv file, enabling access to a variety of sources.  
- **Conversational Retrieval**: The chatbot employs advanced conversational retrieval techniques to deliver relevant, context-aware responses.  
- **Integration of Language Models**: OpenAI's language models are utilized for natural language understanding and generation, allowing the chatbot to engage in meaningful interactions.  
- **CSV Content Extraction**: Text content is extracted from uploaded CSVs, forming the basis for indexing and retrieval.  
- **Text Chunking for Efficiency**: The extracted text is divided into smaller chunks, enhancing retrieval efficiency and ensuring precise answers.  

## Usage

- **Upload CSV File**: Utilize the sidebar to upload CSV file to the application.  
- **Ask Your Questions**: Enter questions in the main chat interface related to the content of the uploaded CSV.  
- **Get Answers**: The chatbot will provide responses based on the information extracted from the CSV.  

## Sample Output
![Output](demos/demo_min.gif)

### WorkFlow
![WorkFlow](workflow.png)

### Query Flow
![Query Flow](queryflow.png)


## Installation

To install and run the app, follow these steps:

Clone the repository 

```
git clone https://github.com/spilabkorea/chatbot.git
```

Add your OpenAI Key:

```
OPENAI_API_KEY=
OPENAI_MODEL_NAME=gpt-4o
OPENAI_EMBEDDING_MODEL_NAME=text-embedding-3-small
```

Create a conda environment

to run this app do activate environment and run app


Install the dependencies using requirements.txt

```bash
pip install -r requirements.txt
```

```
streamlit run app.py
```


