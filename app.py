import streamlit as st

import os
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import re

from collections import Counter

from utils import scrapper

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import random
import time
import json

#-------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "thenlper/gte-large"
READER_MODEL_NAME  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

#vector_database = None

#-------------------------------------------------------------------
def retrieve_articles():
    today = datetime.today()
    yesterday = today - timedelta(days=1)

    # scrapper is already searching for LeMonde.fr in internal
    archive_links = scrapper.create_archive_links(yesterday.year, today.year, 
                                        yesterday.month, today.month, 
                                        yesterday.day, today.day)
    corpus_path = os.path.join(os.getcwd(), "corpus_links")
    scrapper.create_folder(corpus_path)
    article_links = {}
    for year,links in archive_links.items():
        print("processing: ",year)
        article_links_list = scrapper.get_articles_links(links)
        article_links[year] = article_links_list
        scrapper.write_links(corpus_path,article_links_list,year)
    
    themes = []
    for link_list in article_links.values():
        themes.extend(scrapper.list_themes(link_list))
    
    theme_stat = Counter(themes)
    theme_top = []
    for k,v in sorted(theme_stat.items(), key = lambda x:x[1], reverse=True):
        #if v > 700:
        theme_top.append((k, v))
        
    all_links = []
    for link_list in article_links.values():
        all_links.extend(link_list)

    themes_top_five = [x[0] for x in theme_top]

    classified_articles = scrapper.classify_links(themes_top_five,all_links)
    return classified_articles


#-------------------------------------------------------------------
def scrape_articles(articles):
    scrapper.erase_folder_contents('corpus')
    scrapper.create_folder('corpus')

    scrapper.scrape_articles(articles)


#-------------------------------------------------------------------
def split_documents(
    chunk_size: int,
    knowledge_base: List[Document],
    tokenizer,
) -> List[Document]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


#-------------------------------------------------------------------
#--------------------------- INIT -----------------------------------
#-------------------------------------------------------------------
@st.cache_resource
def cache_corpus():
    print('!!! CACHE CORPUS !!!')
    # scrap articles from the last 24h from Le Monde
    articles = retrieve_articles()
    scrape_articles(articles)


#-------------------------------------------------------------------
@st.cache_resource
def get_splitted_documents():
    # load articles with langchain
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = DirectoryLoader("corpus/", glob='**/**/*.txt', loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    
    # split into chunks
    model_max_length = SentenceTransformer(EMBEDDING_MODEL_NAME).max_seq_length
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    
    docs_processed = split_documents(
        model_max_length,
        docs,
        tokenizer)
    
    return docs_processed


#-------------------------------------------------------------------
@st.cache_resource
def get_embedding_model():
    # create vector database with FAISS and embbed
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=False,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )
    return embedding_model


#-------------------------------------------------------------------
@st.cache_resource
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    return tokenizer


#-------------------------------------------------------------------
@st.cache_resource
def get_llm_pipeline():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
    
    tokenizer = get_tokenizer()
    
    llm = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=1500,
        )
    return llm


#-------------------------------------------------------------------
@st.cache_resource
def init():
    print('init')
    
    cache_corpus()
    
    documents = get_splitted_documents()
    embedding_model = get_embedding_model()
    
    vector_database = FAISS.from_documents(
        documents, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    
    if 'vector_database' not in st.session_state:
        st.session_state['vector_database'] = vector_database
    
    get_llm_pipeline()
    
    return


#-------------------------------------------------------------------
def get_rag_prompt_template():
    tokenizer = get_tokenizer()
    
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """
    You are a news reporter summerizing news for the last 24 hours. 
    For each document in context, try to retreive information related to the user's subject, and select document that are relevant.
    If there is no explicit link between a document and the subject, ignore this document.
    If you don't have enought information to make a report based on the subject, say that you can't make a report.
    Then, If you have enought information, make a very detailed news report with the information from the selected documents. 
    Just give facts, don't draw any conclusions.

    Context:
    {context}

    """,
        },
        {
            "role": "user",
            "content": """
    ---
    Now, here are the subjects from which you should build your news reports.

    subject: What are the most important news reguarding {subject} ? """,
        },
    ]

    rag_prompt_template = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    return rag_prompt_template


#-------------------------------------------------------------------
def answer_with_rag(
            subject: str,
            llm: pipeline,
            knowledge_index: FAISS,
            num_retrieved_docs: int = 30,
            num_docs_final: int = 5,
        ) -> Tuple[str, List[Document]]:
    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(query=subject, k=num_retrieved_docs)

    relevant_docs_text = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # # Optionally rerank results
    # if reranker:
    #     print("=> Reranking documents...")
    #     relevant_docs_text = reranker.rerank(subject, relevant_docs_text, k=num_docs_final)
    #     relevant_docs_text = [doc["content"] for doc in relevant_docs_text]

    relevant_docs_text = relevant_docs_text[:num_docs_final]
    relevant_docs = relevant_docs[:num_docs_final]

    #relevant_docs_text = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs_text)])

    rag_prompt_template = get_rag_prompt_template()
    final_prompt = rag_prompt_template.format(subject=subject, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs_text, relevant_docs


#-------------------------------------------------------------------
# Streamed response emulator - TEMP
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


#-------------------------------------------------------------------
def get_source_link_from_relevant_docs(relevant_docs):
    metas = [doc.metadata for doc in relevant_docs]
    sources = [(meta['source']) for meta in metas]
    sources = list(set(sources))

    datas = []
    for source_link in sources:
        meta_link = source_link.replace(".txt", ".meta")
        with open(meta_link, "r") as file:
            loaded_data = json.load(file)
        datas.append(loaded_data)
    return datas

#-------------------------------------------------------------------
#--------------------------- APP -----------------------------------
#-------------------------------------------------------------------

st.set_page_config(
    page_title="RT News Summarizer",
    layout="centered",
    #layout="wide",
    initial_sidebar_state="expanded")

init()

st.title("Realtime News Summarizer")

st.write('Answers will be based on articles from [Le Monde](https://www.lemonde.fr) on the last 24h.')
st.write(f'Current LLM is : [{READER_MODEL_NAME}](https://huggingface.co/{READER_MODEL_NAME})')

st.header('News subjects :')

#print(vector_database)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter a subject you would like news on"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # --- Do magic here ---
    vector_database = st.session_state.vector_database
    #st.write(vector_database)
    
    subject = prompt
    llm = get_llm_pipeline()
    
    answer, relevant_docs_text, relevant_docs = answer_with_rag(subject, llm, vector_database,
                                                                num_retrieved_docs=30, num_docs_final=7)
    answer_no_think = re.sub(r".*?</think>", "", answer, flags=re.DOTALL)
    answer = answer_no_think.lstrip()
    
    metas = get_source_link_from_relevant_docs(relevant_docs)
    
    if metas:
        links_str = "  \n  \nLinks :  \n"
        for meta in metas:
            title = meta["title"]
            link = meta["link"]
            links_str = links_str + f"[{title}]({link})" + "  \n"

        answer = answer + links_str

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write(answer)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})