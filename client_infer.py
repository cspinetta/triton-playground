import numpy as np
from PIL import Image
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import json


def load_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image for ResNet50 inference.
    - Resize to 224x224
    - Normalize to [0,1]
    - Convert HWC to NCHW
    - Add batch dimension
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np.transpose((2, 0, 1))  # HWC → CHW
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dim
    return image_np


def load_class_labels(json_path: str) -> dict:
    """
    Load ImageNet class index mapping from JSON.
    Returns a dict mapping class IDs to readable labels.
    """
    with open(json_path, "r") as f:
        index = json.load(f)
    return {int(k): v[1] for k, v in index.items()}


def run_inference(image_np: np.ndarray) -> int:
    """
    Send a preprocessed image to the Triton server and return predicted class index.
    """
    client = httpclient.InferenceServerClient(url="localhost:8000")

    input_tensor = httpclient.InferInput(
        name="data",
        shape=image_np.shape,
        datatype=np_to_triton_dtype(image_np.dtype)
    )
    input_tensor.set_data_from_numpy(image_np)

    output_tensor = httpclient.InferRequestedOutput("resnetv17_dense0_fwd")

    response = client.infer(
        model_name="resnet50",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )

    output_data = response.as_numpy("resnetv17_dense0_fwd")
    return int(np.argmax(output_data))


if __name__ == "__main__":
    image_path = "sample.jpg"
    labels_path = "imagenet_class_index.json"

    image_np = load_image(image_path)
    predicted_class = run_inference(image_np)
    labels = load_class_labels(labels_path)

    label = labels.get(predicted_class, "Unknown")
    print(f"Predicted class: {label} (ID: {predicted_class})")
