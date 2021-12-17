from flask import Flask, request, redirect, render_template, flash, url_for
from urllib.parse import urlparse

from werkzeug.utils import secure_filename
import os

from io import BytesIO
from PIL import Image
from time import perf_counter

import requests
import logging
import base64

from transformers import (
    VisionEncoderDecoderModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    ViTFeatureExtractor,
)
import torch
import numpy as np

import sqlite3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_poem_from_image(
    vision_encoder_decoder_model,
    vision_encoder_decoder_tokenizer,
    poem_generator,
    poem_tokenizer,
    feature_extractor,
    file_folder,
    filename,
):
    try:
        img = Image.open(os.path.join(file_folder, filename)).convert("RGB")
    except:
        return ""
    try:
        pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values
        description = generate_caption(
            vision_encoder_decoder_model, vision_encoder_decoder_tokenizer, pixel_values
        )
        print(description)
        description = "@" + description[0] + "@"
        generated_text = generate_poem(poem_generator, poem_tokenizer, description)
        # generated_text = generated_text.split("@")[2]
    except:
        return "잘못된 이미지"
    print(generated_text)
    return generated_text


def generate_caption(
    vision_encoder_decoder_model, vision_encoder_decoder_tokenizer, pixel_values
):
    generated_ids = vision_encoder_decoder_model.generate(
        pixel_values.to(device), num_beams=5
    )
    generated_text = vision_encoder_decoder_tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )
    return generated_text


def generate_poem(poem_generator, poem_tokenizer, input_text):
    input_ids = poem_tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = poem_generator.generate(
            input_ids,
            max_length=64,
            repetition_penalty=2.0,
            pad_token_id=poem_tokenizer.pad_token_id,
            eos_token_id=poem_tokenizer.eos_token_id,
            bos_token_id=poem_tokenizer.bos_token_id,
            do_sample=True,
            top_k=30,
            top_p=0.95,
        )
        generated_text = poem_tokenizer.decode(output[0])

    return generated_text
