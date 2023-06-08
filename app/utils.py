from pathlib import Path
import os
from typing import Optional
import uuid
from fastapi import UploadFile
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.base import Embeddings
from joblib import Parallel, delayed
from langchain.text_splitter import TokenTextSplitter
import asyncio
import pickle
import sqlite3
con = sqlite3.connect("filemap.db")
cursor = con.cursor()
from sqlite3 import Connection

DATA_STORE_ROOT = Path(os.environ['HOME']) / 'datastore'
PDF_STORE_ROOT =Path( DATA_STORE_ROOT / 'uploaded_files/')
PDF_STORE_ROOT.mkdir(exist_ok=True,parents=True)

CHROMA_STORE_ROOT = Path(DATA_STORE_ROOT / 'chroma/')
CHROMA_STORE_ROOT.mkdir(exist_ok=True, parents = True)



def save_file_to_collection(filename : str,collection : str, contents: bytes| None):

    try:
        with open(DATA_STORE_ROOT / "collection.index","rb") as f:
            collection_index = pickle.load(f)
    except:
        collection_index = set()

    PARENT_DIR = PDF_STORE_ROOT / collection
    PARENT_DIR.mkdir(exist_ok = True, parents = True)
    OUTPATH = PARENT_DIR / filename
    with open(OUTPATH, 'wb') as f:
        f.write(contents)

    collection_index.add(collection)

    with open(DATA_STORE_ROOT / "collection.index","wb") as f:
        pickle.dump(collection_index,f)


    return OUTPATH

def get_collection_names():
    try:
        with open(DATA_STORE_ROOT / "collection.index","rb") as f:
            collection_index = pickle.load(f)
    except:
        collection_index = set()

    return list(collection_index)


def get_chroma_Store_location(collection_name : str | None = ''):
    
    OUT_PATH = CHROMA_STORE_ROOT / str(collection_name) 
    OUT_PATH.mkdir(exist_ok=True, parents=True)
    return str( OUT_PATH)


async def embed_documents(paths : list[Path | str],collection_name : str, embedding_func :Optional[Embeddings] = None  ):
    vector_store = Chroma(  collection_name=collection_name,
                                embedding_function=embedding_func,
                                persist_directory=get_chroma_Store_location(collection_name)
                                )
    
    load_pdf = lambda x: PyMuPDFLoader(str(x)).load()
    data = Parallel(n_jobs=-1)(delayed(load_pdf)(filepath) for filepath in paths)

    #TODO - track the pdf data and map it to filepath

    data = [doc for file in data for doc in file]

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    pdf_data_as_docs = text_splitter.split_documents(data)

    #TODO - map the chunked docs to filepath

    texts = [doc.page_content for doc in pdf_data_as_docs]
    metadatas = [doc.metadata for doc in pdf_data_as_docs]
    ids = [str(uuid.uuid1()) for _ in texts]


    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    




async def add_documents_to_collection(   collection_name : str,
                                        uploaded_files :list[UploadFile],
                                        con: Connection,
                                        embedding_func:Optional[Embeddings] = None ):
    
    
    contents = await asyncio.gather(*[file.read() for file in uploaded_files])
    
    paths = Parallel(n_jobs=-1)(delayed(save_file_to_collection)(uploaded_files[i].filename, collection_name,contents[i]) for i in range(len(uploaded_files)))

    await embed_documents(paths, collection_name, embedding_func)

    return [file.filename for file in uploaded_files]


async def add_document_to_collection(   collection_name : str,
                                        uploaded_file : UploadFile,
                                        con: Connection,
                                        embedding_func:Optional[Embeddings] = None ):
    vector_store = Chroma(  collection_name=collection_name,
                                embedding_function=embedding_func,
                                persist_directory=get_chroma_Store_location(collection_name)
                                )
    
    #load contents
    contents = await uploaded_file.read()
    
    # first save the PDF
    PDF_PATH = save_file_to_collection(uploaded_file.filename,collection_name, contents)

    #Extract list of documents from PDF
    data = PyMuPDFLoader(str(PDF_PATH)).load()
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    pdf_data_as_docs = text_splitter.split_documents(data)
    texts = [doc.page_content for doc in pdf_data_as_docs]
    metadatas = [doc.metadata for doc in pdf_data_as_docs]
    ids = [str(uuid.uuid1()) for _ in texts]

    # # Save the idx <-> pdf mapping in an sqlite3 db
    # cursor = con.cursor()
    # cursor.execute("CREATE TABLE IF NOT EXISTS pdf2id(id varchar(20), filename varchar(100), collection varchar(100))")
    # for idx in ids:
    #     cursor.execute(f"INSERT INTO pdf2id VALUES (\"{idx}\", \"{uploaded_file.filename}\", \"{collection_name}\")")

    # con.commit()

    # res = cursor.execute("SELECT * FROM pdf2id")
    # print(res.fetchall())
    # Add the data to the vectorstore
    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
