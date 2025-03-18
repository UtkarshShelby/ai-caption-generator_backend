from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io

app = FastAPI()

# ✅ Enable CORS so frontend can communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load BLIP-2 Model and Processor (Check if CUDA is available)
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
@app.get("/")  # ✅ Add this route
def read_root():
    return {"message": "Backend is running!"}
# ✅ API Route to Generate Captions
@app.post("/generate_caption/")
async def generate_caption(file: UploadFile = File(...)):  # Corrected

    try:
        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Process image and generate caption
        inputs = processor(image, return_tensors="pt").to(device)  # Ensure processing happens on GPU if available
        caption_ids = model.generate(**inputs)
        caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]

        return {"caption": caption}

    except Exception as e:
        return {"error": str(e)}
