

from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import requests
import json
from gradio_client import Client
import os

pitch_extraction_method = Literal["pm", "harvest", "crepe", "rmvpe"]

F0_CURVE_PATH = "shared/f0/f0G48k.pth"
@dataclass
class GradioClientInfo:
    url: str
    client: Optional[Client] = None
    busy: bool = False


def to_relative_path(original_path: str, base_dir: str) -> str:
    path_parts = original_path.split(os.sep)
    
    try:
        index = path_parts.index(base_dir)
    except ValueError:
        raise ValueError(f"'{base_dir}' not found in the path")
    
    relative_path = '.' + os.sep + os.sep.join(path_parts[index:])
    return relative_path

class InferenceParams(BaseModel):
    transpose_pitch: int
    pitch_extraction_method: str
    search_feature_ratio: float
    filter_radius: int
    audio_resampling: int
    volume_envelope_scaling: float
    artifact_protection: float

    def __post_init__(self):
        if not 0.0 <= self.search_feature_ratio <= 1.0:
            raise ValueError("search_feature_ratio must be an float from 0.0 to 1.0")
        if round(self.search_feature_ratio, 2) != self.search_feature_ratio:
            self.search_feature_ratio = round(self.search_feature_ratio, 2)

        if not 0 <= self.filter_radius <= 7:
            raise ValueError("filter_radius must be an integer from 0 to 7")

        if not 0 <= self.audio_resampling <= 48000:
            raise ValueError("audio_resampling must be an integer from 0 to 48000")

        if not 0.0 <= self.volume_envelope_scaling <= 1.0:
            raise ValueError("volume_envelope_scaling must be an float from 0.0 to 1.0")
        if round(self.volume_envelope_scaling, 2) != self.volume_envelope_scaling:
            self.volume_envelope_scaling = round(self.volume_envelope_scaling, 2)

        if not 0.0 <= self.artifact_protection <= 0.5:
            raise ValueError("artifact_protection must be an float from 0.0 to 0.5")
        if round(self.artifact_protection, 2) != self.artifact_protection:
            self.artifact_protection = round(self.artifact_protection, 2)


# 'shared/logs/b6eb29566a524703f9bf850851b6b3ca2bb471184abb75253e6b50fd60aa9aef.index'
# 'shared/logs/e0a71032233030619d4f9d05ffe2cde4fdb2cd97b2ae7aeaa1fe459df4d7b646.index'
def infer_convert(
    client: Client,
    model_weight_filename: str,
    model_index_path: str,
    f0_curve_path: str,
    audio_input_file: str,
    params: InferenceParams,
) -> str:
    # Load the model weight
    client.predict(
        model_weight_filename,
        0,  # TODO: Figure out how to remove this since it gets set again for inference (float between 0 and 0.5) Protect voiceless consonants)
        0,  # TODO: Figure out how to remove this since it gets set again for inference (float between 0 and 0.5) Protect voiceless consonants)
        api_name="/infer_change_voice",
    )

    # Run inference
    result = client.predict(
        0,  # float (numeric value between 0 and 2333) in 'Select Speaker/Singer ID:' Slider component
        audio_input_file,  # str  in 'Enter the path of the audio file to be processed (default is the correct format example):' Textbox component
        params.transpose_pitch,  # float  in 'Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12):' Number component
        F0_CURVE_PATH,  # str (filepath on your computer (or URL) of file) in 'F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation:' File component
        params.pitch_extraction_method,  # str  in 'Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement' Radio component
        model_index_path,  # str  in 'Path to the feature index file. Leave blank to use the selected result from the dropdown:' Textbox component
        model_index_path,  # str (Option from: ['logs/added_IVF163_Flat_nprobe_1_Omni-Man_MK1_v2.index']) in 'Auto-detect index path and select from the dropdown:' Dropdown component
        params.search_feature_ratio,  # float (numeric value between 0 and 1) in 'Search feature ratio (controls accent strength, too high has artifacting):' Slider component
        params.filter_radius,  # float (numeric value between 0 and 7) in 'If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.' Slider component
        params.audio_resampling,  # float (numeric value between 0 and 48000) in 'Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling:' Slider component
        params.volume_envelope_scaling,  # float (numeric value between 0 and 1) in 'Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume:' Slider component
        params.artifact_protection,  # float (numeric value between 0 and 0.5) in 'Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy:' Slider component
        api_name="/infer_convert",
    )

    audio_output_path = result[1]
    audio_output_path = to_relative_path(
        original_path=audio_output_path, base_dir="shared"
    )
    return audio_output_path