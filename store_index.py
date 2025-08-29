from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
import os
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data = 'Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings() 

pc = PineconeGRPC(api_key=PINECONE_API_KEY)

index_name = "mrdoctor"

# Create index
pc.create_index(
    name=index_name,
    dimension=384,   # must match your embedding model dimension
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)


docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)