import streamlit as st
import pytesseract
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
import torch

# Load pre-trained DETR model for object detection
model_name = "facebook/detr-resnet-50"
image_processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# Define class names for COCO dataset
class_names = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Function to classify the subject of the photo
def detect_objects(image):
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Filter out predictions with low scores
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detected_objects = []
    for score, label in zip(results["scores"], results["labels"]):
        detected_objects.append(f"{class_names[label]}: {score:.2f}")

    return detected_objects

# Function to detect text in the image
def detect_text(image):
    return pytesseract.image_to_string(image)

# Streamlit app layout
st.title("Image Subject and Text Detection")
st.write("Upload an image to identify the subject and detect any text.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Display the original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect objects in the image
    detected_objects = detect_objects(image)
    st.markdown(f"**Detected Objects:** {', '.join(detected_objects)}")

    # Detect text in the image
    detected_text = detect_text(image)
    st.markdown(f"**Detected Text:** {detected_text}")
