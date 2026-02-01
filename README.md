# Diplom_project_AI
AI system for analyzing blood biochemistry test results from scanned documents
[https://docs.google.com/presentation/d/1hqippMqa0iTMANDGa7M8JCmxqwbCzx-jw2WVHpYG40g/edit?usp=sharing](https://docs.google.com/presentation/d/1UqgSyNQXwNCRndk3kXlJyYR-plS7yjdbzWpLHjCnjCE/edit?usp=sharing)

Создание ИИ системы, которая предназначена для автоматического распознавания показателей из результатов биохимических анализов крови, представленных в виде цифровых изображений, с целью упрощения извлечения данных и минимизации ошибок.


# Blood Test OCR & Analysis (FastAPI)

Web application for recognizing biochemical blood test results from scanned images or PDFs.

## Features
- Upload blood test (PDF / JPG / PNG)
- YOLOv8 detects table area
- EasyOCR extracts text hints
- GPT-4o interprets results
- Clean text output
- Simple web UI (FastAPI + HTML/CSS)

## Tech Stack
- Python
- FastAPI
- YOLOv8 (Ultralytics)
- EasyOCR
- OpenAI GPT-4o

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
