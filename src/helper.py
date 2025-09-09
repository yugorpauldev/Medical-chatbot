from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document 


#Extract Data from the PDF file

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf",
                             loader_cls=PyPDFLoader)
  
    documents = loader.load()
    return documents 


#Filer the document to just keep the source and the text 
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("Source", "")  # Default to an empty string if "Source" is None or missing
        if src is None:
            src = ""  # Ensure the value is a string
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

#Split the text into chunks

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


#Download embedding model

def download_embedding():
    
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
        
    )
    return embeddings