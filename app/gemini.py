import os
import json
from io import BytesIO
from PIL import Image, ImageDraw
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Your Google API Key should be stored as GOOGLE_API_KEY in your .env
API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Initialize the Gemini client once (for bounding-box detection and chat).
# You can re-initialize for different models or usage scenarios if needed.
client = genai.Client(api_key=API_KEY)

# System instructions for bounding-box detection
BOUNDING_BOX_SYSTEM_INSTRUCTIONS = """
Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Identify all the objects in the scenario. Do not miss any object.
If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
"""

# For additional safety. Adjust to your needs or remove if undesired.
SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

# ---------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------

def _parse_json(json_output: str) -> str:
    """
    Strip away any markdown fences (```json ... ```) from the model’s response,
    so we can load it properly as JSON.
    """
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i+1:])
            if "```" in json_output:
                json_output = json_output.split("```")[0]
            break
    return json_output

def _draw_bounding_boxes(original_image: Image.Image, bounding_boxes_str: str) -> Image.Image:
    """
    Given an image and a JSON string containing bounding boxes in the format:
    [
      {
        "label": "some_label",
        "box_2d": [y1, x1, y2, x2]  (normalized in 0..1000)
      },
      ...
    ]

    Returns a copy of the image with bounding boxes drawn on it.
    """
    # Parse any markdown fences from the model’s response
    bounding_boxes_str = _parse_json(bounding_boxes_str)
    
    # Attempt to load bounding box data as JSON
    try:
        bounding_boxes = json.loads(bounding_boxes_str)
    except json.JSONDecodeError:
        # If the model returned something non-JSON, just return the original image
        return original_image.copy()

    # Create a copy of the original image so we don't modify the source
    out_img = original_image.copy()
    draw = ImageDraw.Draw(out_img)

    # Some sample colors for bounding boxes
    colors = [
        "red", "green", "blue", "yellow", "orange", "pink", "purple",
        "brown", "gray", "cyan", "magenta", "lime"
    ]

    width, height = out_img.size

    for idx, box in enumerate(bounding_boxes):
        color = colors[idx % len(colors)]
        box_2d = box.get("box_2d", [])

        if len(box_2d) == 4:
            # Gemini returns [y1, x1, y2, x2] in [0..1000] normalized coords
            y1_norm, x1_norm, y2_norm, x2_norm = box_2d
            y1 = int((y1_norm / 1000) * height)
            x1 = int((x1_norm / 1000) * width)
            y2 = int((y2_norm / 1000) * height)
            x2 = int((x2_norm / 1000) * width)

            # Ensure coordinates are in the correct order
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # Draw the bounding box
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

            # Label the box
            label = box.get("label", f"obj_{idx}")
            draw.text((x1 + 5, y1 + 5), label, fill=color)

    return out_img

# ---------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------

def generate_masked_image(pil_image: Image.Image, prompt: str = "Detect all objects"):
    """
    Generates a bounding-box overlay image using Gemini 2.0’s object detection.
    If you want to “mask out” other parts of the image, you can adapt this
    to fill the background, etc.

    Returns:
        A new PIL.Image with bounding boxes drawn on it.
    """
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in environment variables.")

    # Convert prompt to something that references bounding boxes
    final_prompt = prompt

    # Use the gemini-2.0-flash model (best for bounding box detection).
    # If you want to parameterize the model name, you can add an argument.
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[final_prompt, pil_image],
        config=types.GenerateContentConfig(
            system_instruction=BOUNDING_BOX_SYSTEM_INSTRUCTIONS,
            temperature=0.5,
            safety_settings=SAFETY_SETTINGS,
        )
    )

    # Draw bounding boxes
    masked_image = _draw_bounding_boxes(pil_image, response.text)
    return masked_image

def request_gemini_chat(
    model_name: str,
    user_message: str,
    pil_image: Image.Image = None,
    conversation_history: list = None,
    system_instructions: str = "You are a helpful assistant. Please provide useful responses by making questions about the image given until the user is satisfied.",
):
    """
    Send a message (and optionally an image) to a Gemini model for multi-turn conversation.

    Args:
        model_name: The name of the Gemini model to call (e.g., "gemini-2.0-pro", "gemini-1.5-pro", etc.)
        user_message: The user’s text message for the model.
        pil_image: An optional PIL image to provide context (e.g., the masked image).
        conversation_history: A list of previous messages for multi-turn conversation.
        system_instructions: The system prompt to guide the model’s overall behavior.

    Returns:
        The model’s response (string).
    """
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in environment variables.")

    # Build the content array
    # If you have a multi-turn conversation, you can join them or keep them separate
    # as multiple text inputs. For example:
    # contents = [sys_prompt, user1, user2, ... , user_message, pil_image?]
    # The new Google GenAI client generally expects [text, optional images, optional attachments].
    # This is a minimal approach—feel free to adapt for your conversation style.

    conversation_text = ""
    if conversation_history:
        # Example: each item in conversation_history is a dict { "role": "user"/"assistant", "content": ... }
        for turn in conversation_history:
            conversation_text += f"{turn['role'].capitalize()}: {turn['content']}\n"

    conversation_text += f"User: {user_message}\n"

    contents = [conversation_text]
    if pil_image is not None:
        contents.append(pil_image)

    # Make the request
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instructions,
            temperature=0.7,  # adjust as needed
            safety_settings=SAFETY_SETTINGS,
        )
    )

    return response.text
