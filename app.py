import os
import io
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
from dotenv import load_dotenv # Import dotenv


# Import necessary classes from your original script / transformers
from transformers import (
    SwinModel,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForCausalLM, # Added for Llama
    AutoTokenizer,       # Added for Llama
)
from transformers.modeling_outputs import BaseModelOutput

load_dotenv() # Load environment variables from .env file

# --- Configuration ---
MODEL_PATH = '/cluster/home/ammaa/Downloads/Projects/CheXpert-Report-Generation/swin-t5-model.pth'  # Path to your trained model weights
SWIN_MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
T5_MODEL_NAME = "t5-base"
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Llama model
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") # Get token from env

if not HF_TOKEN:
    print("Warning: HUGGING_FACE_HUB_TOKEN environment variable not set. Llama model download might fail.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_FOLDER = 'uploads' # Optional: If you want to save uploads temporarily
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists if you use it
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# --- Swin-T5 Model Definition ---
class ImageCaptioningModel(nn.Module):
    def __init__(self,
                 swin_model_name=SWIN_MODEL_NAME,
                 t5_model_name=T5_MODEL_NAME):
        super().__init__()
        self.swin = SwinModel.from_pretrained(swin_model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.img_proj = nn.Linear(self.swin.config.hidden_size, self.t5.config.d_model)

    def forward(self, images, labels=None):
        swin_outputs = self.swin(images)
        img_feats = swin_outputs.last_hidden_state
        img_feats_proj = self.img_proj(img_feats)
        encoder_outputs = BaseModelOutput(last_hidden_state=img_feats_proj)
        if labels is not None:
            outputs = self.t5(encoder_outputs=encoder_outputs, labels=labels)
        else:
            outputs = self.t5(encoder_outputs=encoder_outputs)
        return outputs

# --- Global Variables for Model Components ---
swin_t5_model = None
swin_t5_tokenizer = None
transform = None
llama_model = None
llama_tokenizer = None

def load_swin_t5_model_components():
    """Loads the Swin-T5 model, tokenizer, and transformation pipeline."""
    global swin_t5_model, swin_t5_tokenizer, transform
    try:
        print(f"Loading Swin-T5 model components on device: {DEVICE}")
        # Initialize model structure
        swin_t5_model = ImageCaptioningModel(swin_model_name=SWIN_MODEL_NAME, t5_model_name=T5_MODEL_NAME)

        # Load state dictionary
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"Swin-T5 Model file not found at {MODEL_PATH}.")
        # Load Swin-T5 model to the primary DEVICE (can be CPU or GPU)
        swin_t5_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        swin_t5_model.to(DEVICE)
        swin_t5_model.eval()  # Set to evaluation mode
        print("Swin-T5 Model loaded successfully.")

        # Load tokenizer
        swin_t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
        print("Swin-T5 Tokenizer loaded successfully.")

        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print("Transforms defined.")

    except Exception as e:
        print(f"Error loading Swin-T5 model components: {e}")
        raise

def load_llama_model_components():
    """Loads the Llama model and tokenizer."""
    global llama_model, llama_tokenizer
    if not HF_TOKEN:
        print("Skipping Llama model load: Hugging Face token not found.")
        return # Don't attempt to load if no token

    try:
        print(f"Loading Llama model ({LLAMA_MODEL_NAME}) components...")
        # Use bfloat16 for memory efficiency if available, otherwise float16/32
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, token=HF_TOKEN)
        llama_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map="auto", # Automatically distribute across GPUs/CPU RAM if needed
            token=HF_TOKEN
            # Add quantization config here if needed (e.g., load_in_4bit=True with bitsandbytes)
            # quantization_config=BitsAndBytesConfig(...)
        )
        llama_model.eval() # Set to evaluation mode
        print("Llama Model and Tokenizer loaded successfully.")

    except Exception as e:
        print(f"Error loading Llama model components: {e}")
        # Decide if the app should run without the chat feature or crash
        llama_model = None
        llama_tokenizer = None
        print("WARNING: Chatbot functionality will be disabled due to loading error.")
        # raise # Uncomment this if the chat feature is critical

# --- Inference Function (Swin-T5) ---
def generate_report(image_bytes, selected_vlm, max_length=100):
    """Generates a report/caption for the given image bytes using Swin-T5."""
    global swin_t5_model, swin_t5_tokenizer, transform
    if not all([swin_t5_model, swin_t5_tokenizer, transform]):
        # Check if loading failed or wasn't called
        if swin_t5_model is None or swin_t5_tokenizer is None or transform is None:
             load_swin_t5_model_components() # Attempt to load again if missing
             if not all([swin_t5_model, swin_t5_tokenizer, transform]):
                 raise RuntimeError("Swin-T5 model components failed to load.")
        else:
             raise RuntimeError("Swin-T5 model components not loaded properly.")


    if selected_vlm != "swin_t5_chexpert":
        return "Error: Selected VLM is not supported."

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension and send to device

        # Perform inference
        with torch.no_grad():
            swin_outputs = swin_t5_model.swin(input_image)
            img_feats = swin_outputs.last_hidden_state
            img_feats_proj = swin_t5_model.img_proj(img_feats)
            encoder_outputs = BaseModelOutput(last_hidden_state=img_feats_proj)

            generated_ids = swin_t5_model.t5.generate(
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            report = swin_t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return report

    except Exception as e:
        print(f"Error during Swin-T5 report generation: {e}")
        return f"Error generating report: {e}"

# --- Chat Function (Llama 3.1) ---
def generate_chat_response(question, report_context, max_new_tokens=250):
    """Generates a chat response using Llama based on the report context."""
    global llama_model, llama_tokenizer
    if not llama_model or not llama_tokenizer:
        return "Chatbot is currently unavailable."

    # System prompt to guide the LLM
    system_prompt = "You are a helpful medical assistant. I'm a medical student, your task is to help me understand the following report."
    # Construct the prompt using the chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Based on the following report:\n\n---\n{report_context}\n---\n\nPlease answer this question: {question}"}
    ]

    # Prepare input for the model
    try:
        # Use the tokenizer's chat template
        input_ids = llama_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(llama_model.device) # Move input IDs to the same device as the model

        # Set terminators for generation
        # Common terminators for Llama 3 Instruct
        terminators = [
            llama_tokenizer.eos_token_id,
            llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with torch.no_grad():
            outputs = llama_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True, # Use sampling for more natural responses
                temperature=0.6,
                top_p=0.9,
                pad_token_id=llama_tokenizer.eos_token_id # Avoid warning, set pad_token_id
            )

        # Decode the response, skipping the input prompt part
        response_ids = outputs[0][input_ids.shape[-1]:]
        response_text = llama_tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_text.strip()

    except Exception as e:
        print(f"Error during Llama chat generation: {e}")
        return f"Error generating chat response: {e}"


# --- Flask Application Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load models when the application starts
print("Loading models on application startup...")
try:
    load_swin_t5_model_components()
    load_llama_model_components() # Load Llama
    print("Model loading complete.")
except Exception as e:
    print(f"FATAL ERROR during model loading: {e}")
    # Depending on requirements, you might want to exit or continue with limited functionality
    # exit(1) # Example: Exit if models are critical

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---- NEW: Function to Parse Filename ----
def parse_patient_info(filename):
    """
    Parses a filename like '00069-34-Frontal-AP-63.0-Male-White.png'
    Returns a dictionary with 'view', 'age', 'gender', 'ethnicity'.
    Returns None if parsing fails.
    """
    try:
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('-')
        # Expected structure based on example: ... - ViewPart1 - ViewPartN - Age - Gender - Ethnicity
        if len(parts) < 5: # Need at least initial parts, age, gender, ethnicity
            print(f"Warning: Filename '{filename}' has fewer parts than expected.")
            return None

        ethnicity = parts[-1]
        gender = parts[-2]
        age_str = parts[-3]
        # Handle potential '.0' in age and convert to int
        try:
            age = int(float(age_str))
        except ValueError:
            print(f"Warning: Could not parse age '{age_str}' from filename '{filename}'.")
            return None # Or set age to None/default

        # Assume view is everything between the second part (index 1) and the age part (index -3)
        view_parts = parts[2:-3]
        view = '-'.join(view_parts) if view_parts else "Unknown" # Handle cases with missing view

        # Basic validation
        if gender.lower() not in ['male', 'female', 'other', 'unknown']: # Be flexible
             print(f"Warning: Unusual gender '{gender}' found in filename '{filename}'.")
             # Decide whether to return None or keep it

        return {
            'view': view,
            'age': age,
            'gender': gender.capitalize(), # Capitalize for display
            'ethnicity': ethnicity.capitalize() # Capitalize for display
        }
    except IndexError:
        print(f"Error parsing filename '{filename}': Index out of bounds.")
        return None
    except Exception as e:
        print(f"Error parsing filename '{filename}': {e}")
        return None

# --- Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    chatbot_available = bool(llama_model and llama_tokenizer)
    return render_template('index.html', chatbot_available=chatbot_available)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and prediction."""
    chatbot_available = bool(llama_model and llama_tokenizer) # Check again
    patient_info = None # Initialize patient_info

    if 'image' not in request.files:
        flash('No image file part in the request.', 'danger')
        return redirect(url_for('index'))

    file = request.files['image']
    vlm_choice = request.form.get('vlm_choice', 'swin_t5_chexpert')
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

            # ---- ADDED: Parse filename ----
            original_filename = file.filename
            patient_info = parse_patient_info(original_filename)
            if patient_info:
                print(f"Parsed Patient Info: {patient_info}")
            else:
                print(f"Could not parse patient info from filename: {original_filename}")
            # ---- END ADDED ----

            # Generate report using Swin-T5
            report = generate_report(image_bytes, vlm_choice, max_length)

            # Check for errors in report generation
            if report.startswith("Error"):
                 flash(f'Report Generation Failed: {report}', 'danger')
                 # Still render with image if possible, but show error
                 image_data = base64.b64encode(image_bytes).decode('utf-8')
                 return render_template('index.html',
                                        report=None, # Or pass the error message
                                        image_data=image_data,
                                        patient_info=patient_info, # Pass parsed info even if report failed
                                        chatbot_available=chatbot_available)


            image_data = base64.b64encode(image_bytes).decode('utf-8')

            # Render the page with results AND the report text for JS/Chat
            return render_template('index.html',
                                   report=report,
                                   image_data=image_data,
                                   patient_info=patient_info, # Pass the parsed info
                                   chatbot_available=chatbot_available) # Pass availability again

        except FileNotFoundError as fnf_error:
             flash(f'Model file not found: {fnf_error}. Please check server configuration.', 'danger')
             print(f"Model file error: {fnf_error}\n{traceback.format_exc()}")
             return redirect(url_for('index'))
        except RuntimeError as rt_error:
            flash(f'Model loading error: {rt_error}. Please check server logs.', 'danger')
            print(f"Runtime error during prediction (model loading?): {rt_error}\n{traceback.format_exc()}")
            return redirect(url_for('index'))
        except Exception as e:
            flash(f'An unexpected error occurred during prediction: {e}', 'danger')
            print(f"Error during prediction: {e}\n{traceback.format_exc()}")
            return redirect(url_for('index'))
    else:
        flash('Invalid image file type. Allowed types: png, jpg, jpeg.', 'danger')
        return redirect(url_for('index'))

# --- New Chat Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests based on the generated report."""
    if not llama_model or not llama_tokenizer:
        return jsonify({"answer": "Chatbot is not available."}), 503 # Service unavailable

    data = request.get_json()
    if not data or 'question' not in data or 'report_context' not in data:
        return jsonify({"error": "Missing question or report context"}), 400

    question = data['question']
    report_context = data['report_context']

    try:
        answer = generate_chat_response(question, report_context)
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({"error": "Failed to generate chat response"}), 500

if __name__ == '__main__':
    # Make sure to set debug=False for production/sharing
    app.run(host='0.0.0.0', port=5000, debug=False)