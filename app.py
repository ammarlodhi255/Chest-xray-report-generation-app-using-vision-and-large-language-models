import os
import io
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, flash, redirect, url_for

# Import necessary classes from your original script / transformers
from transformers import (
    SwinModel,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from transformers.modeling_outputs import BaseModelOutput

# --- Configuration ---
MODEL_PATH = '/cluster/home/ammaa/Downloads/Projects/CheXpert-Report-Generation/swin-t5-model.pth'  # Path to your trained model weights
SWIN_MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
T5_MODEL_NAME = "t5-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_FOLDER = 'uploads' # Optional: If you want to save uploads temporarily
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists if you use it
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# --- Model Definition (Copied from your script) ---
class ImageCaptioningModel(nn.Module):
    def __init__(self,
                 swin_model_name=SWIN_MODEL_NAME,
                 t5_model_name=T5_MODEL_NAME):
        super().__init__()
        self.swin = SwinModel.from_pretrained(swin_model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.img_proj = nn.Linear(self.swin.config.hidden_size, self.t5.config.d_model)

    def forward(self, images, labels=None):
        # This forward is primarily for training/loss calculation.
        # For inference, we usually call components separately or use generate.
        swin_outputs = self.swin(images)
        img_feats = swin_outputs.last_hidden_state
        img_feats_proj = self.img_proj(img_feats)
        encoder_outputs = BaseModelOutput(last_hidden_state=img_feats_proj)
        if labels is not None:
            outputs = self.t5(encoder_outputs=encoder_outputs, labels=labels)
        else:
            # For inference without labels, T5 generate method is typically used externally
            # This path might not be directly used in our inference function below
            outputs = self.t5(encoder_outputs=encoder_outputs)
        return outputs

# --- Global Variables for Model Components ---
# Load model and tokenizer globally on startup to avoid reloading per request
model = None
tokenizer = None
transform = None

def load_model_components():
    """Loads the model, tokenizer, and transformation pipeline."""
    global model, tokenizer, transform
    try:
        print(f"Loading model components on device: {DEVICE}")
        # Initialize model structure
        model = ImageCaptioningModel(swin_model_name=SWIN_MODEL_NAME, t5_model_name=T5_MODEL_NAME)

        # Load state dictionary
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please place the '.pth' file correctly.")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully.")

        # Load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
        print("Tokenizer loaded successfully.")

        # Define image transformations (should match training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print("Transforms defined.")

    except Exception as e:
        print(f"Error loading model components: {e}")
        # Handle error appropriately - maybe raise it to stop Flask if model is essential
        raise

# --- Inference Function ---
def generate_report(image_bytes, selected_vlm, max_length=100):
    """Generates a report/caption for the given image bytes."""
    global model, tokenizer, transform
    if not all([model, tokenizer, transform]):
        raise RuntimeError("Model components not loaded properly.")

    # Basic check for VLM choice - expand if more models added
    if selected_vlm != "swin_t5_chexpert":
        return "Error: Selected VLM is not supported."

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension and send to device

        # Perform inference
        with torch.no_grad():
            # 1. Get image features from Swin
            swin_outputs = model.swin(input_image)
            img_feats = swin_outputs.last_hidden_state

            # 2. Project features
            img_feats_proj = model.img_proj(img_feats)

            # 3. Wrap features for T5 encoder input
            encoder_outputs = BaseModelOutput(last_hidden_state=img_feats_proj)

            # 4. Generate text using T5 decoder
            generated_ids = model.t5.generate(
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                num_beams=4,        # Beam search parameters (can be tuned)
                early_stopping=True
            )

            # 5. Decode generated IDs to text
            report = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return report

    except Exception as e:
        print(f"Error during report generation: {e}")
        return f"Error generating report: {e}"


# --- Flask Application Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flashing messages

# Load model when the application starts
load_model_components()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and prediction."""
    if 'image' not in request.files:
        flash('No image file part in the request.', 'danger')
        return redirect(url_for('index'))

    file = request.files['image']
    vlm_choice = request.form.get('vlm_choice', 'swin_t5_chexpert') # Get selected VLM
    try:
        max_length = int(request.form.get('max_length', 100))
        if not (10 <= max_length <= 512):
            raise ValueError("Max length must be between 10 and 512.")
    except ValueError as e:
         flash(f'Invalid Max Length value: {e}', 'danger')
         return redirect(url_for('index'))


    if file.filename == '':
        flash('No image selected for uploading.', 'warning')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            image_bytes = file.read()
            # Generate report
            report = generate_report(image_bytes, vlm_choice, max_length)

            # Encode image bytes to base64 to display on the results page without saving
            image_data = base64.b64encode(image_bytes).decode('utf-8')

            # Render the page again with results
            return render_template('index.html', report=report, image_data=image_data)

        except Exception as e:
            flash(f'An error occurred: {e}', 'danger')
            print(f"Error during prediction: {e}") # Log the full error server-side
            return redirect(url_for('index'))
    else:
        flash('Invalid image file type. Allowed types: png, jpg, jpeg.', 'danger')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network
    # debug=True is useful for development but should be OFF in production
    app.run(host='0.0.0.0', port=5003, debug=False)