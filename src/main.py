from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from fastapi import FastAPI, HTTPException, Response, File, Form, UploadFile, Request
from fastapi.responses import FileResponse
from dataclasses import dataclass
from dotenv import load_dotenv
import os
from gradio_client import Client
from gradio_helpers import GradioClientInfo, InferenceParams, infer_convert
import asyncio
import shutil
from tempfile import NamedTemporaryFile
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import json

GRADIO_SERVER_URLS = [
    "http://localhost:7865/",
    "http://localhost:7866/",
    "http://localhost:7867/",
]

RVC_OUT_DIR = "shared/output"

load_dotenv()
PORT = int(os.getenv("PORT"))

clients: List[GradioClientInfo] = []

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)


@app.exception_handler(RequestValidationError)
def initialize_clients():
    for url in GRADIO_SERVER_URLS:
        client = Client(url, output_dir=RVC_OUT_DIR)
        clients.append(GradioClientInfo(url=url, client=client, busy=False))


async def get_available_client() -> Optional[GradioClientInfo]:
    while True:
        for client_info in clients:
            if not client_info.busy:
                client_info.busy = True
                return client_info
        await asyncio.sleep(0.05)  # Wait a bit before trying again


class VoiceConvertArgs(BaseModel):
    inference_params: InferenceParams
    weights_sha256: str
    f0_curve: str


@app.post("/voice_convert")
async def voice_convert(
    args: str = Form(...),
    audio: UploadFile = File(...),
):
    args = json.loads(args)

    if args == None:
        raise HTTPException(status_code=400, detail="args is required")

    gradio_client = await get_available_client()
    model_weights_filename = f"{args['weights_sha256']}.pth"
    model_index_path = f"shared/logs/{args['weights_sha256']}.index"
    f0_curve_path = f"shared/f0/{args['f0_curve']}"

    try:
        with NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name
            input_file_path = shutil.copy(tmp_path, "shared/input")

        inference_params_dict = args["inference_params"]
        inference_params = InferenceParams(
            transpose_pitch=inference_params_dict["transpose_pitch"],
            pitch_extraction_method=inference_params_dict["pitch_extraction_method"],
            search_feature_ratio=inference_params_dict["search_feature_ratio"],
            filter_radius=inference_params_dict["filter_radius"],
            audio_resampling=inference_params_dict["audio_resampling"],
            volume_envelope_scaling=inference_params_dict["volume_envelope_scaling"],
            artifact_protection=inference_params_dict["artifact_protection"],
        )
        audio_output_file = infer_convert(
            gradio_client.client,
            model_weights_filename,
            model_index_path,
            f0_curve_path,
            input_file_path,
            inference_params,
        )

    finally:
        if input_file_path and os.path.isfile(input_file_path):
            os.remove(input_file_path)

        gradio_client.busy = False

        if audio_output_file and os.path.isfile(audio_output_file):
            with open(audio_output_file, 'rb') as source_audio, NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                shutil.copyfileobj(source_audio, tmp)
                tmp_path = tmp.name

                directory_to_remove = os.path.dirname(audio_output_file)
                shutil.rmtree(directory_to_remove)

                return FileResponse(
                    tmp_path,
                    media_type="audio/wav",
                    filename="speech.wav",
                    headers={"Content-Disposition": "inline; filename=speech.wav"},
                )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the error details
    logging.error(f"Error for request {request}: {exc}")
    # You can also include more information in the response if needed
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


if __name__ == "__main__":
    import uvicorn

    initialize_clients()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
