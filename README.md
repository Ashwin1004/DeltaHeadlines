# Image-to-PDF to Summarized Audio Conversion Project

This project involves converting JPG images into PDF format, extracting text from the PDF, summarizing the extracted text using an advanced AI model, and finally converting the summarized text into audio using a text-to-speech (TTS) model.

## Project Workflow
1. **Image to PDF Conversion**
2. **Text Extraction from PDF**
3. **Text Summarization Using DeepScaleR-1.5B-Preview**
4. **Text-to-Speech Conversion Using gTTS**

---

## Step 1: Image to PDF Conversion

For converting JPG images to PDF format, the `img2pdf` Python library was used. This ensures image quality preservation and facilitates better text extraction.

**Code Example:**
```python
import img2pdf
from PIL import Image
import os

def convert_images_to_pdf(image_folder, output_pdf):
    images = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(".jpg")]
    with open(output_pdf, "wb") as pdf_file:
        pdf_file.write(img2pdf.convert(images))

convert_images_to_pdf("images", "output.pdf")
```

---

## Step 2: Text Extraction from PDF

For extracting text from the generated PDF, `PyPDF2` was utilized.

**Code Example:**
```python
import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extract_text()
    return text

pdf_text = extract_text_from_pdf("output.pdf")
print(pdf_text)
```

---

## Step 3: Text Summarization Using DeepScaleR-1.5B-Preview

The summarization model used is `agentica-org/DeepScaleR-1.5B-Preview`. This model is efficient due to its deep learning architecture optimized for scale, allowing it to handle lengthy texts better than common summarizers like `facebook/bart-cnn`. While BART excels at news-like content, DeepScaleR is specifically designed to manage larger contexts and complex language structures, making it ideal for our use case.

**Code Example:**
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("agentica-org/DeepScaleR-1.5B-Preview")
model = AutoModelForSeq2SeqLM.from_pretrained("agentica-org/DeepScaleR-1.5B-Preview")

def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=4096, truncation=True)
    summary_ids = model.generate(**inputs, max_length=200, min_length=50, length_penalty=2.0)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

summary = summarize_text(pdf_text)
print(summary)
```

## **Sentence transformer Architecture**

![WhatsApp Image 2025-03-13 at 09 50 33_51cc6713](https://github.com/user-attachments/assets/ea96368c-ef88-4efa-96a8-1bb5d894957a)


---

## Step 4: Text-to-Speech Conversion Using gTTS

For converting the summarized text to speech, the `gTTS` library was employed. It is lightweight, fast, and effective for generating natural-sounding speech from text.

**Code Example:**
```python
from gtts import gTTS

def text_to_speech(text, output_audio):
    tts = gTTS(text)
    tts.save(output_audio)

text_to_speech(summary, "output_audio.mp3")
```

---

## Requirements
Install the following libraries before running the project:
```
pip install img2pdf PyPDF2 transformers gtts
```

---

## Usage
1. Place your `.jpg` images in the `images/` folder.
2. Run the `convert_images_to_pdf()` function to create a PDF.
3. Extract text from the PDF using `extract_text_from_pdf()`.
4. Summarize the extracted text using `summarize_text()`.
5. Convert the summarized text to speech using `text_to_speech()`.

---

## Conclusion
This project effectively integrates multiple tools and models for streamlined document processing. The use of `DeepScaleR-1.5B-Preview` ensures high-quality summarization, outperforming models like `facebook/bart-cnn` for complex text data. Combined with `gTTS`, this project provides a comprehensive pipeline from visual data to audio output.



https://github.com/user-attachments/assets/13695d20-26b8-494b-b3c3-432fdfb2c107


https://github.com/user-attachments/assets/a5cc5db1-20d7-41e0-94d5-b2c1f337b7f7


