import os
import boto3
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock

def hr_index():

    data_load=PyPDFLoader('https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf')

    # data_test=data_load.load_and_split()
    # print(len(data_test))
    # print(data_test[2])

    data_split=RecursiveCharacterTextSplitter(separators=["\n\n","\n"," ",""],chunk_size=100,chunk_overlap=10)
    # "\n\n" means first split into a paragraph
    #  "\n" means then split into a lines
    # " " character 
    #  "" so on
    data_embeddings= BedrockEmbeddings(
        model_id='amazon.titan-embed-text-v1'
    )

    data_index=VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS
    )

    db_index=data_index.from_loaders([data_load])
    return db_index


def hr_llm():
    llm=Bedrock(
    model_id='anthropic.claude-v2',
    model_kwargs={
        "max_tokens_to_sample":300,
        "temperature":0.1,
        "top_p":0.9,
    }
    )
    return llm

def hr_rag_response(index,question):
    rag_llm=hr_llm()
    hr_rag_query=index.query(question=question,llm=rag_llm)
    return hr_rag_query