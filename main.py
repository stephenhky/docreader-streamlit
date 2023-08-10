
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from summarizer import TransformerSummarizer


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=20,
    length_function=len
)

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

hub = st.radio('openai', ['openai', 'huggingface-langchain', 'huggingface-native'])
if hub == 'openai':
    llm_model_name = st.radio('gpt-3.5-turbo', ['gpt-3.5-turbo', 'gpt-4'])
    temperature = st.number_input('temperature', min_value=0.0, max_value=1.0, value=0.7)
    max_tokens = st.number_input('max_tokens', min_value=1, value=600)
    llm_model = ChatOpenAI(temperature=temperature, model_name=llm_model_name, max_tokens=max_tokens)
    embeddings = OpenAIEmbeddings()
elif hub == 'huggingface-langchain':
    llm_model_name = st.radio(
        'gpt2',
        ['gpt2', 'gpt2-large', 'google/flan-t5-xxl', 'databricks/dolly-v2-3b']
    )
    temperature = st.number_input('temperature', min_value=0.0, max_value=1.0, value=0.7)
    llm_model = HuggingFaceHub(repo_id=llm_model_name, model_kwargs={'temperature': temperature})
    embedding = st.radio(
        'gpt2',
        ['gpt2', 'gpt2-large',
         'sentence-transformers/all-MiniLM-L6-v2',
         "sentence-transformers/allenai-specter"
         'google/flan-t5-xxl', 'databricks/dolly-v2-3b']
    )

    embeddings = HuggingFaceEmbeddings(model_name=embedding)
    if embeddings.client.tokenizer.pad_token is None:
        embeddings.client.tokenizer.pad_token = embeddings.client.tokenizer.eos_token
elif hub == 'huggingface-native':
    transformer_type = st.radio(
        'GPT2',
        ['GPT2', 'XLNet']
    )
    if transformer_type == 'GPT2':
        llm_model_name = st.radio(
            'gpt2',
            ['gpt2', 'gpt2-medium', 'gpt2-large', 'distilgpt2']
        )
    elif transformer_type == 'XLNet':
        llm_model_name = st.radio(
            'xlnet-base-cased',
            ['xlnet-base-cased', 'xlnet-large-cased', 'textattack/xlnet-base-cased-imdb']
        )
    min_length = st.number_input('min_length', min_value=1, value=60)
    summarizer = TransformerSummarizer(transformer_type='GPT2', transformer_model_key=llm_model_name)
else:
    pass

uploaded_pdffile = st.file_uploader('Upload a file (.pdf)')

st.text('What to do?')
action = st.radio('Summarize', ['Summarize', 'Retrieve'])

if (uploaded_pdffile is not None):
    pdfbytes = tempfile.NamedTemporaryFile()
    tempfilename = pdfbytes.name
    pdfbytes.write(uploaded_pdffile.read())

    loader = PyPDFLoader(tempfilename)
    pages = loader.load_and_split(text_splitter=text_splitter)

    if action == 'Summarize':
        if st.button('Summarize'):
            if hub in ['openai', 'huggingface-langchain']:
                chain = load_summarize_chain(llm=llm_model, chain_type='map_reduce')
                response = chain.run(pages)

                st.markdown(response)
            elif hub == 'huggingface-native':
                body = ' '.join([page.page_content for page in pages])
                summary = summarizer(body, min_length=min_length)

                st.markdown(summary)

    elif action == 'Retrieve':
        if hub in ['openai', 'huggingface-langchain']:
            db = FAISS.from_documents(pages, embeddings)

            question = st.text_area('Ask a question:')
            to_ask = st.button('Ask')
            if to_ask:
                retriever = db.as_retriever()
                qa = RetrievalQA.from_chain_type(
                    llm=llm_model,
                    chain_type='stuff',
                    retriever=retriever,
                    return_source_documents=False
                )
                answer = qa({'query': question})
                st.markdown(answer['result'])
        else:
            st.text('Not implemented.')