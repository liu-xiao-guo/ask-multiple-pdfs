import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from htmlTemplates import css, bot_template, user_template

import lib_indexer
import lib_llm
import lib_embeddings
import lib_vectordb

index_name = "pdf_docs"

def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_pdf_texts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # chunks = text_splitter.split_documents(text)
    return chunks

def get_text_chunks1(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks
        
def handle_userinput(db, llm_chain_informed, user_question):
    similar_docs = db.similarity_search(user_question)
    print(f'The most relevant passage: \n\t{similar_docs[0].page_content}')

    ## 4. Ask Local LLM context informed prompt
    # print(">> 4. Asking The Book ... and its response is: ")
    informed_context= similar_docs[0].page_content
    response = llm_chain_informed.run(context=informed_context,question=user_question)

    st.write(user_template.replace("{{MSG}}", user_question).replace("{{MSG1}}", " "), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", response).replace("{{MSG1}}", similar_docs[0].page_content),unsafe_allow_html=True)
            
def main():
    
    # # Huggingface embedding setup
    hf = lib_embeddings.setup_embeddings()

    # # # ## Elasticsearch as a vector db
    db, url = lib_vectordb.setup_vectordb(hf, index_name)

    # # # ## set up the conversational LLM
    llm_chain_informed= lib_llm.make_the_llm()

    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        handle_userinput(db, llm_chain_informed, user_question)
            
    st.write(user_template.replace("{{MSG}}", "Hello, human").replace("{{MSG1}}", " "), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello, robot").replace("{{MSG1}}", " "), unsafe_allow_html=True)
    
    # Add a side bar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and press on click on Process", accept_multiple_files=True)
        print(pdf_docs)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get pdf text from
                # raw_text = get_pdf_text(pdf_docs[0])
                raw_text = get_pdf_texts(pdf_docs)
                # st.write(raw_text)
                print(raw_text)
                
                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                
                # Create vector store
                lib_indexer.loadPdfChunks(text_chunks, url, hf, db, index_name)


if __name__ == "__main__":
    main()