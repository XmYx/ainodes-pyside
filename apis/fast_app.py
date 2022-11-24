import logging
import time

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


from backend.singleton import singleton

gs = singleton


from apis.v1 import txt2img_api



app = FastAPI()

origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f'{process_time:0.4f} sec')
    return response

logging.basicConfig(level=logging.DEBUG)   # add this line
logger = logging.getLogger("foo")


app.include_router(txt2img_api.router)


#app.mount("/", StaticFiles(directory="www/web/static", html=True), name="static")

uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
