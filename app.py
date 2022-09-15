from fastapi import FastAPI
import uvicorn
from transformers import AutoTokenizer
import onnxruntime as ort
from utils import GenerativePagasus
import time

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers=['CUDAExecutionProvider']
tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
decoder_sess = ort.InferenceSession(str("onnx_pagasus/pagasus-decoder-with-lm-head.onnx"),sess_options=so, providers=providers)
encoder_sess = ort.InferenceSession(str("onnx_pagasus/pagasus-encoder.onnx"),sess_options=so, providers=providers)
generative_pagasus = GenerativePagasus(encoder_sess, decoder_sess, tokenizer, onnx=True,cuda=True)

app = FastAPI()

@app.get('/')
async def home():
    return {'ParaPhrase':'Go to the /paraphrase to get the paraphrase'}
    
@app.get("/paraphrase")
async def paraphrase(body: str,max_len: int = 20, temp: float = 0.):
    start = time.time()
    generated = generative_pagasus(body, max_len, temperature=temp)[0]
    end = time.time() - start
    spt = end/len(generated.split())
    print("Second per Token: ",spt)
    out = {'Output':generated,'Second/token':spt}
    return out

if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0',port=5000)
