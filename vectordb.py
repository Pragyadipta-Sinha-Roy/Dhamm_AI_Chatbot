import os
import yaml
from pyprojroot import here
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine


class PrepareVectorDB:

    def __init__(self, file_path, chunk_size, chunk_overlap, embedding_model, vectordb_dir, collection_name):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.collection_name = collection_name


    def validate_file(self) -> bool:
        """
        Validates if the provided file exists and is a text file.

        Returns:
            bool: True if file exists and is a text file, False otherwise
        """
        if not os.path.exists(here(self.file_path)):
            print(f"Error: File '{self.file_path}' does not exist.")
            return False
        
        if not self.file_path.lower().endswith(('.txt', '.text')):
            print(f"Error: File '{self.file_path}' is not a text file.")
            return False
        
        return True

    def run(self) -> None:
        """
        Executes the main logic to create and store document embeddings in a VectorDB.

        If the vector database directory doesn't exist:
        - It loads the text document from file_path, splits it into chunks
        - Embeds the document chunks using the specified embedding model
        - Stores the embeddings in a persistent VectorDB directory

        If the directory already exists, it skips the embedding creation process.

        Returns:
            None
        """
        # Validate input file
        if not self.validate_file():
            return

        if not os.path.exists(here(self.vectordb_dir)):
            # Create the directory and embeddings
            os.makedirs(here(self.vectordb_dir))
            print(f"Directory '{self.vectordb_dir}' was created.")

            # Load the text file
            loader = TextLoader(str(here(self.file_path)), encoding='utf-8')
            docs_list = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            doc_splits = text_splitter.split_documents(docs_list)

            # Add to vectorDB
            vectordb = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding=GoogleGenerativeAIEmbeddings(model=self.embedding_model),
                persist_directory=str(here(self.vectordb_dir))
            )
            print("VectorDB is created and saved.")
            print(f"Number of vectors in vectordb: {vectordb._collection.count()}\n")
        else:
            print(f"Directory '{self.vectordb_dir}' already exists.")


if __name__ == "__main__":
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    prepare_db_instance = PrepareVectorDB(
        file_path=r"C:\Users\KIIT0001\Documents\ML project\Dhamm_AI_Chatbot\cleaned_transcript.txt",
        chunk_size=5000,
        chunk_overlap=1000,
        embedding_model='models/embedding-001',
        vectordb_dir='vectordb',
        collection_name='chroma'
    )
    
    prepare_db_instance.run()