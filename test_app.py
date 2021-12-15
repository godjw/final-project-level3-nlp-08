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
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    PreTrainedTokenizerFast,
)
import torch
import numpy as np

import sqlite3


# TODO: 이미지이름 한글로 되어있을 때 처리해야됨.
# TODO: PNG 만 받게 처리?
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])
UPLOAD_FOLDER = "web/assets/uploads"

app = Flask(__name__, static_url_path="", static_folder="", template_folder="web/templates")
app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename=UPLOAD_FOLDER + "/" + filename), code=301)


@app.route("/")
def index():
    print(urlparse(request.url))
    # filename = request.args.get("filename")
    if request.args.get("filename"):
        filename = request.args.get("filename").split("/")[-1]
        # print(request.args.get("generated_poem"))
        generated_poem = temp_generate(filename)
        # if filename and request.args.get("generated_poem"):
        return render_template(
            "responsive.html",
            filename=filename,
            generated_poem=generated_poem,
        )
    else:
        return render_template("responsive.html")


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":

        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No image selected for uploading")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(
                # os.path.join(app.config["STATIC_FOLDER"], os.path.join(app.config["UPLOAD_FOLDER"], filename))
                os.path.join(app.config["UPLOAD_FOLDER"], filename)
            )
            # print('upload_image filename: ' + filename)
            flash("Image successfully uploaded and displayed below")
            return render_template("responsive.html", filename=filename)
        else:
            flash("Allowed image types are -> png, jpg, jpeg, gif")
            return redirect(request.url)

    elif request.method == "GET":
        # print(urlparse(request.url))
        filename = request.args.get("filename")
        generated_poem = request.args.get("generated_poem")

        return render_template(
            "responsive.html",
            filename=request.args.get("filename"),
            generated_poem=request.args.get("generated_poem"),
        )


def generate_poem(input_text):
    input_ids = poem_tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        # Check generation time
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

        # print("generated text:", generated_text, sep="\n")

    return generated_text


def captioning(pixel_values):

    generated_ids = model.generate(pixel_values.to(device), num_beams=5)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_text


# @app.route("/" + static + "filename" + "generated_poem", methods=["GET"])
# def display_all():
#     return render_template("responsive_all.html")


def temp_generate(filename):
    try:
        img = Image.open(os.path.join(app.config["UPLOAD_FOLDER"], filename)).convert("RGB")
    except:
        return ""
    try:
        pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values
        description = captioning(pixel_values)
        generated_text = generate_poem(description[0])
    except:
        return ""
    return generated_text


@app.route("/generate/<filename>", methods=["GET", "POST"])
def generate(filename):
    if request.method == "GET":
        t = perf_counter()
        content = request.json

        # if content:
        #     img_url = request.json.get("img_url", "")
        #     try:
        #         img = Image.open(requests.get(img_url, stream=True).raw)
        #     except:
        #         return "올바르지 않은 url입니다."
        # else:
        #     img = Image.open(request.files["img"]).convert("RGB")

        # if not img:
        #     return ""
        if content:
            # filename = content.get('filename', "")
            # file = request.files["file"]
            # filename = secure_filename(file.filename)
            print(filename)
            try:
                img = Image.open(os.path.join(app.config["UPLOAD_FOLDER"], filename)).convert("RGB")
                # pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values
                # description = captioning(pixel_values)
                # generated_poem = generate_poem(description[0])
            except:
                return "잘못된 이미지"

        # try:
        #     pixel_values = feature_extractor(
        #         images=img, return_tensors="pt"
        #     ).pixel_values
        #     description = captioning(pixel_values)
        #     generated_text = generate_poem(description[0])
        # except:
        #     return "잘못된 이미지"

        print("time: ", (perf_counter() - t))

        # TODO: input 형식 확인
        # request: image
        ## 1) url -> image store -> load -> model input
        ##### image -> wget image_url -> db -> db image PIL Open
        ##### os.system('wget', url)
        ## 2) image binary -> model input

        # metadatas
        # metadata -> request args parsing (json, ...)

        # prompt 랑 같이 생성할지 아닐지, 폰트, 배경이미지 등 조절할지아닐지

        # model = load_model()

        # input_ids = model(image).input_ids
        # tokenizer.decode(input_ids)

        # response: string
        #     if phase == "sentiment":
        #         return redirect(url_fol("http://localhost/generate/sentiment"))
        #     else:
        #         generated_string = model.generate(pixel_values=image)

        #         #TODO: db에 생성한 시 저장하기

        #         return generated_string

        # elif request.method == "GET":
        #     return "이미지를 POST 형식으로 보내주세요"

        # buffer = BytesIO()
        # img.save(buffer, format="jpeg")
        # imgbytes = buffer.getvalue()

        # conn = sqlite3.connect("db.db")
        # cursor = conn.cursor()

        # client_id = "ashhhhhdf"
        # generated_text = "aaaa"

        # cursor.execute("insert or IGNORE into CLIENT (ID) values (?)", [client_id])
        # cursor.execute(
        #     "insert into  POEM (CLIENT_ID, IMG, POEM) values (?, ?, ?)",
        #     (client_id, imgbytes, generated_text),
        # )
        # conn.commit()

        generated_poem = "눈앞에 아른아르고 있다."

        # return render_template("responsive.html", filename=filename, generated_poem=generated_poem)
        return redirect(
            url_for("upload_image", filename=UPLOAD_FOLDER + "/" + filename, generated_poem=generated_poem),
            code=301,
        )
        # return generated_text


# TODO
# 1. 기존 api 연결
# 1.1 model 로딩
# 1.2 버튼 event 연결
# 1.3 generate 한 시를 html template 으로 보내주는작업
# 2. db 연결
# - 2.1 이미지를 db에 저장
# 3. html 편집 및 가독성 확장


if __name__ == "__main__":
    poem_generator = AutoModelForCausalLM.from_pretrained(
        "CheonggyeMountain-Sherpa/kogpt-trinity-poem", use_auth_token=True
    )
    poem_tokenizer = AutoTokenizer.from_pretrained(
        "CheonggyeMountain-Sherpa/kogpt-trinity-poem", use_auth_token=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    poem_generator.to(device)
    poem_generator.eval()
    print("capitoning_model load")
    # device setting

    # load feature extractor and tokenizer
    encoder_model_name_or_path = "ddobokki/vision-encoder-decoder-vit-gpt2-coco-ko"
    feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_model_name_or_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(encoder_model_name_or_path)

    # load model
    model = VisionEncoderDecoderModel.from_pretrained(encoder_model_name_or_path)
    model.to(device)
    print("generator model load")

    app.run(host="0.0.0.0", port=6006, debug=True)
