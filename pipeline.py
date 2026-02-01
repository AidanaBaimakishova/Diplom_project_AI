import os
import cv2
import base64
from ultralytics import YOLO
from pdf2image import convert_from_path
import easyocr
from dotenv import load_dotenv
from openai import OpenAI

from synonyms import get_synonyms_text
from config import YOLO_MODEL_PATH, TEMP_DIR, CONF_THRESHOLD

load_dotenv()
print("USING KEY:", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

os.makedirs(TEMP_DIR, exist_ok=True)

_reader = None
_model = None


def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["ru"])
    return _reader


def get_model():
    global _model
    if _model is None:
        _model = YOLO(YOLO_MODEL_PATH)
    return _model


def encode_image_b64(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")


def load_images(file_path):
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path)
        paths = []
        for i, page in enumerate(pages):
            path = os.path.join(TEMP_DIR, f"{os.path.basename(file_path)}_{i}.jpg")
            page.save(path, "JPEG")
            paths.append(path)
        return paths
    return [file_path]


def detect_and_crop(image_path):
    img = cv2.imread(image_path)
    model = get_model()
    results = model(image_path)

    crops = []
    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            crops.append(crop)
    return crops


def ocr_hint(image):
    reader = get_reader()
    texts = reader.readtext(image)
    return " ".join([t[1] for t in texts])


from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def send_to_llm(image, hint):
    img_b64 = encode_image_b64(image)
    synonyms_text = get_synonyms_text()

    prompt = f"""
Ты помощник для распознавания биохимических анализов крови.

Основной источник истины — изображение.
OCR-подсказка может содержать ошибки, используй её только как помощь.

СИНОНИМЫ ПОКАЗАТЕЛЕЙ:
{synonyms_text}

Верни результат обычным ТЕКСТОМ (не JSON) строго в формате:

Биохимический анализ крови
Название показателя        Значение
Ферритин                   0
Железо (Fe)                0
...

Правила:
- один показатель = одна строка
- использовать канонические названия
- ничего лишнего не добавлять

OCR-подсказка:
{hint}
"""

    response = client.responses.create(
        model="gpt-4o",
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{img_b64}"
                }
            ]
        }],
        max_output_tokens=1000
    )

    return response.output_text


def process_file(file_path):
    images = load_images(file_path)
    texts = []

    for img_path in images:
        for crop in detect_and_crop(img_path):
            hint = ocr_hint(crop)
            text = send_to_llm(crop, hint)
            texts.append(text)

    if not texts:
        return "Ничего не распознано на изображении"

    return "\n\n".join(texts)



