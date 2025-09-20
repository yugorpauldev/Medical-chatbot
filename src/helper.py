from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document 
import os



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
        # Extract the source and page number from metadata
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        
        # Get the book name from the source path and clean it up
        book_name = os.path.basename(source) if source else "Unknown"
        
        # Clean the book title more thoroughly
        import re
        # Remove file extension
        clean_book_name = os.path.splitext(book_name)[0]
        # Replace hyphens and underscores with spaces
        clean_book_name = clean_book_name.replace('-', ' ').replace('_', ' ')
        # Remove long number sequences (like ISBNs or file hashes)
        clean_book_name = re.sub(r'\b\d{10,}\b', '', clean_book_name)
        # Remove common file suffixes like 'oss'
        clean_book_name = re.sub(r'\b(oss|pdf)\b', '', clean_book_name, flags=re.IGNORECASE)
        # Clean up extra spaces
        clean_book_name = ' '.join(clean_book_name.split()).strip()
        
        # Handle specific known books for better formatting
        if 'laboratory' in clean_book_name.lower() and 'hematology' in clean_book_name.lower():
            clean_book_name = "A Laboratory Guide to Clinical Hematology"
        elif 'human' in clean_book_name.lower() and 'physiology' in clean_book_name.lower():
            clean_book_name = "Human Physiology"

        # Create a new Document with updated metadata
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": source,
                    "page": page,
                    "book": clean_book_name
                }
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