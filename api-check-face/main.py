# Bibliotecas necessárias
from fastapi import FastAPI, File
from fastapi.responses import JSONResponse
import io
import os
from starlette.responses import StreamingResponse
from tensorflow.keras.models import load_model
from scripts.functions import load_model_and_predict_image
from pydantic import BaseModel
import base64

# Instanciação do FastAPI
app = FastAPI()

# Peso do modelo treinado com Redes Neurais convolucionais
model = load_model('weights/trained_model4.h5')

# Definindo a classe PredictionResponse
class PredictionResponse(BaseModel):
    result: str         # Atributo que representa o resultado da predição (string)
    img_original: str   # Atributo que representa o caminho da imagem original (string)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/face_real_or_fake')
async def predict(img_bytes: bytes = File(...)):
    """ Endpoint que recebe uma imagem do tipo bytes e carrega o peso do modelo treinado e gera uma resposta
    se a imagem é real ou falsa"""

    # Path da imagem que deseja salvar temporariamente
    image_path = "images_temp/imagem.jpg"

    image = io.BytesIO(img_bytes)
    image.seek(0)

    # Salvar a imagem no caminho temporário
    with open(image_path, 'wb') as f:
        f.write(image.getbuffer())

    result = load_model_and_predict_image('images_temp/imagem.jpg', 'weights/trained_model4.h5')

    # Remover a imagem no caminho temporário
    os.remove(image_path)

    return JSONResponse(content={"result": result})

@app.post('/image_sent')
async def image(img_bytes: bytes = File(...)):
    """ Endpoint que recebe uma imagem do tipo bytes e apenas plota no tela"""

    image = io.BytesIO(img_bytes)
    image.seek(0)

    # Imagem Original
    img_original = StreamingResponse(
        image,
        media_type="image/jpg",
    )

    return img_original


@app.post('/face_real_or_fake_and_image_base64', response_model=PredictionResponse)
async def predict_and_send_image(img_bytes: bytes = File(...)):
    """ Endpoint que recebe uma imagem do tipo bytes e carrega o peso do modelo treinado e
    gera uma resposta se a imagem é real ou falsa e gera a imagem em base64 para validação.

    Converte BASE64 em BYTES: https://base64.guru/converter/decode/image"""

    # Path da imagem salva na pasta de arquivos temporarios
    image_path = "images_temp/imagem.jpg"

    image = io.BytesIO(img_bytes)
    image.seek(0)

    # Salvar a imagem no caminho temporário
    with open(image_path, 'wb') as f:
        f.write(image.getbuffer())

    result = load_model_and_predict_image('images_temp/imagem.jpg', 'weights/trained_model4.h5')

    # Imagem original
    with open(image_path, 'rb') as img_file:
        img_original_content = img_file.read()

    # Remover a imagem temporária após o predict
    os.remove(image_path)

    # Gerar imagem base64 e incluir no response
    encoded_img_original = base64.b64encode(img_original_content).decode('utf-8')

    return {"result": result, "img_original": encoded_img_original}


# @app.post('/image_sent', response_class=FileResponse)
# async def image_sent(img_bytes: bytes = File(...)):
#     return io.BytesIO(img_bytes)

# @app.get('/get_image')
# async def get_image():
#     # Assuming you have a 'path_to_image' variable with the correct path to the image file
#     path_to_image = "images_temp/imagem.jpg"
#
#     return FileResponse(path_to_image, media_type="image/jpg")
