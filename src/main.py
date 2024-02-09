from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from fastapi import FastAPI, HTTPException, Response, File, UploadFile
from fastapi.responses import FileResponse
from dataclasses import dataclass
from dotenv import load_dotenv
import os
from gradio_client import Client
from gradio_helpers import GradioClientInfo, InferenceParams, infer_convert
import asyncio
import shutil
from tempfile import NamedTemporaryFile

GRADIO_SERVER_URLS = [
    "http://rvc1:7864/",
    "http://rvc0:7865/",
    "http://rvc2:7867/",
]

RVC_OUT_DIR = "shared/output"

load_dotenv()
PORT = int(os.getenv('PORT'))

clients: List[GradioClientInfo] = []

app = FastAPI()

# client0 = Client(GRADIO_SERVER_URLS[0], output_dir=RVC_OUT_DIR)

client1 = Client(GRADIO_SERVER_URLS[1], output_dir=RVC_OUT_DIR)


# def initialize_clients():
#     for url, index in GRADIO_SERVER_URLS:
#         clients.append(GradioClientInfo(url=url, client=Client(url, output_dir=RVC_OUT_DIR), busy=False))


async def get_available_client() -> Optional[GradioClientInfo]:
    while True:
        for client_info in clients:
            if not client_info.busy:
                client_info.busy = True
                return client_info
        await asyncio.sleep(.05)  # Wait a bit before trying again


@app.post("/voice_convert")
async def voice_convert(audio: UploadFile = File(...), inference_params: InferenceParams = None, weights_sha256: str = None, f0_curve: str = None) -> Response:
    if inference_params == None:
        raise HTTPException(
            status_code=400, detail="inference_params is required")
    if weights_sha256 == None:
        raise HTTPException(
            status_code=400, detail="weights_sha256 is required")
    if f0_curve == None:
        raise HTTPException(status_code=400, detail="f0_curve is required")

    gradio_client = await get_available_client()
    model_weights_filename = f"{weights_sha256}.pth"
    model_index_path = f"shared/logs/{weights_sha256}.index"
    f0_curve_path = f"shared/f0/{f0_curve}"

    try:
        with NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name

        audio_output_file = infer_convert(
            gradio_client,
            model_weights_filename,
            model_index_path,
            f0_curve_path,
            audio,
            inference_params,
        )

    finally:
        if tmp_path and os.path.isfile(tmp_path):
            os.remove(tmp_path)
        gradio_client.busy = False

    return FileResponse(
        tmp_path,
        media_type="audio/wav",
        filename="speech.wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"}
    )


if __name__ == "__main__":
    import uvicorn
    initialize_clients()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
