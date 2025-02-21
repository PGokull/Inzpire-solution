import base64
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

def convert_image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")  # Convert to JPEG format
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def perform_inference(image_path):
    image_base64 = convert_image_to_base64(image_path)

    prompt_text = (
        "Analyze the given image and provide a structured JSON response.\n\n"
        "### Steps:\n"
        "1. Identify the **main object** in the image (e.g., glass, Rangoli, or other).\n"
        "2. If a **glass** is detected, determine if it is **clean** or **dirty** and provide a confidence percentage.\n"
        "3. If **Rangoli** is detected, check if it is **present** or **not** and provide a confidence percentage.\n"
        "4. If the object is **neither a glass nor Rangoli**, return `'object_detected': 'other'`.\n\n"
        "### Strict JSON Output Format:\n"
        "{\n"
        '  "object_detected": "glass / rangoli / other",\n'
        '  "glass_state": "clean or dirty (if applicable)",\n'
        '  "glass_confidence": confidence_percentage (if applicable),\n'
        '  "rangoli_present": "yes or no (if applicable)",\n'
        '  "rangoli_confidence": confidence_percentage (if applicable)\n'
        "}"
    )

    image_part = {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
    text_part = {"type": "text", "text": prompt_text}
    
    message = [HumanMessage(content=[image_part, text_part])]

    llm = ChatOllama(model="llava:latest", temperature=0)
    parser = JsonOutputParser()
    chain = llm | parser
    response = chain.invoke(message)
    
    return response

image_path = r"D:\Inzpire-Solutions\aimodel\testimages\8.jpg"
result = perform_inference(image_path)

print("Inference Result:", result)
print(type(result))