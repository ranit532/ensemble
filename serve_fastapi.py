from fastapi import FastAPI
from utils.io_utils import latest_joblib
import os
app = FastAPI()
@app.get('/health')
def health():
    return {'status':'ok'}
@app.get('/model')
def model_info():
    m = latest_joblib(os.path.join(os.path.dirname(__file__),'models'))
    return {'model': os.path.basename(m) if m else None}
