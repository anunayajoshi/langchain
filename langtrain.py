import os
from dotenv import load_dotenv

load_dotenv()


def train(url_list):

    urls = url_list

    from langchain.document_loaders import UnstructuredURLLoader
    loaders = UnstructuredURLLoader(urls)
    data = loaders.load()

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = text_splitter.split_documents(data)

    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    import pinecone

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_API_ENV'))
    index_name = "langchain2"

    docsearch = Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)

    return docsearch

    

