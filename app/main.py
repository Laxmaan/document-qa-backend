
import logging
import chromadb
from chromadb.config import Settings


from typing import Annotated, Union

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain, LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sqlite3
import json
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT, CONDENSE_QUESTION_PROMPT
from collections import defaultdict
from time import time
from pydantic import BaseModel
from app.utils import PDF_STORE_ROOT
import openai
from loguru import logger

con = sqlite3.connect("filemap.db")

from app.utils import add_document_to_collection, add_documents_to_collection, get_chroma_Store_location
app = FastAPI()
templates = Jinja2Templates(directory="templates")

from app.callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from app.query_data import get_chain
from app.schemas import ChatResponse


### Setup Azure 
import os
# os.environ["OPENAI_API_TYPE"] = openai.api_type = "azure"
# os.environ["OPENAI_API_VERSION"] = openai.api_version = "2022-12-01" #"2023-03-15-preview"
# os.environ["OPENAI_API_BASE"] = openai.api_base =  "https://idocopenaigpt.openai.azure.com/"
# os.environ["OPENAI_API_KEY"] = openai.api_key = "95776649ac7a4b048c834003fd315264"#os.getenv("AZUREOPENAI_API_KEY")



# #### Langchain and Chroma setup 

# chroma_client = chromadb.Client)

# embedding_func = OpenAIEmbeddings( deployment="idocembedd")

embedding_func = OpenAIEmbeddings()


# chromadb_settings = Settings(chroma_api_impl="rest",
#                                         chroma_server_host="localhost",
#                                         chroma_server_http_port="6000"
#                                     )


### CORS
origins = ["*"]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_headers = ["*"], allow_methods=["*"], allow_credentials=True)

###

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)

logger.add(LOG_DIR / "backend_{time}.log", rotation="12:00", enqueue=True)

@app.get("/list")
def list_files():
    logger.info("GOT LIST")
    
    DATA_DIR = PDF_STORE_ROOT
    res = {}
    try:
        for x in DATA_DIR.glob("*"):
            print(x)
            if x.is_dir():
                for item in x.glob("*.pdf"):
                    if x.name not in res:
                        res[x.name] = []
                    res[x.name].append(item.name)
    except Exception as e:
        print(e)
        logger.exception(e)
        res['status'] = 'failure'
        res['message'] = str(e)
        logger.exception(e)

    logger.info(res)
    return res


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

@app.get("/upload/", response_class=HTMLResponse)
async def upload(request: Request):
   return templates.TemplateResponse("upload_file.html", {"request": request})

@app.post("/files/")
async def create_file(file: Annotated[bytes | None, File()] = None):
    if not file:
        return {"message": "No file sent"}
    else:
        return {"file_size": len(file)}

class userLogin(BaseModel):
    username: str
    password: str

@app.post("/user/login")
async def login_check(userdetails: userLogin):
    username = userdetails.username # since outh works on username field we are considering email as username
    password = userdetails.password 

    if username == "arun.krishnan@ilink-systems.com" and password == "Welcome@1": 
        return {"Message":"Success"
                }
    
    else:
        return {"Error":"Invalid User"}
    
@app.post("/uploadfile/")
async def create_upload_file( collection: Annotated[str, Form()],files:Annotated[list[UploadFile] , File(description="Multiple files as UploadFile")] = None):
    start = time()
    collection_name = collection if collection else 'default'
    collection_name = (''.join(filter(lambda x: x.isalnum(), collection_name))).lower()
    retval = {}
    if not files:
        end = time()
        return {"message": "No upload files sent","status":"fail", "time":end - start}
    else:
        retval['filenames'] = []
        retval["collection"] = collection_name
        retval['status'] = 'success'
        try:
            # for file in files:
            #     await add_document_to_collection(collection_name,file,con,embedding_func)
            #     retval['filenames'].append(file.filename)

            retval['filenames'] = await add_documents_to_collection(collection_name,files,con,embedding_func)
            
            end = time()
            retval['time'] = end - start

        except Exception as e:
            retval['status'] = 'fail'
            retval['message'] = f"Error :\n\n {e}"
            
        
        
        return retval




@app.websocket("/chat/{collection}")
@logger.catch
async def chat_over_collection(websocket: WebSocket, collection: str):
    logger.info(f" Chatting over {collection}")
    await websocket.accept()
    try:
        question_handler = QuestionGenCallbackHandler(websocket)
        stream_handler = StreamingLLMCallbackHandler(websocket)
        chat_history = []
        vectorstore =  Chroma(  collection_name=collection,
                    embedding_function=embedding_func,
                    persist_directory=get_chroma_Store_location(collection)
                    )
        qa_chain = get_chain(vectorstore, question_handler, stream_handler)
        
  

        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            logger.info(f"Received data ={data}")
            question = data['question']
            seq = int(data['seq'])
            resp = ChatResponse(sender="you", message=question, type="stream", seq = seq)
            logger.info(resp)
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start",seq=seq)
            logger.info("Begin bot message")
            logger.debug(start_resp)
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end",seq=seq)
            logger.debug(end_resp)
            await websocket.send_json(end_resp.dict())
        

    except WebSocketDisconnect:
            logger.exception("websocket disconnect")
            
    except Exception as e:
        logger.exception(e)
        resp = ChatResponse(
                sender="bot",
                message=f"Sorry, something went wrong. Try again. \n\n{e}",
                type="error",
                seq = -2
            )
        await websocket.send_json(resp.dict())
        return {"status":"fail", "message":str(e)}