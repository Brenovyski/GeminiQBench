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
    system_instructions: str = '''SUMMARY: The user has an objective and you need to find the object in the image that the user wants. 
    You are a helpful assistant. Your goal here is to determine what object the user wants by extracting unknow information from the user to solve ambiguity about the image given and provide helpful responses
    to the user. You can ask questions, provide information, or engage in a conversation history. You can also ask the user to provide more information or clarify their request.
    Try to obtain the unknow information from the user and solve what the user wants. 
    IMPORTANT: After you determined the final answer write in front of it: 'Final answer: ' alongside with the coordinates of the object in the image 
    and then refuses to answer any more questions. The coordinates must be answered in a way that is easy to understand for the user. Avoid numerical coordinates.
    ''',
):
    """
    Build a multi-turn conversation by passing the entire history to the model each time.
    conversation_history is a list of dicts: [{"role": "user"/"assistant", "content": ...}, ...]
    """

    if conversation_history is None:
        conversation_history = []

    # 1) Build the conversation text from history
    #    Each turn is appended as "User: ..." or "Assistant: ..."
    conversation_contents = []
    for turn in conversation_history:
        if turn["role"] == "assistant":
            conversation_contents.append(f"{turn['content']}")
        else:  # user
            conversation_contents.append(f"{turn['content']}")

    # 2) Add the current user message
    conversation_contents.append(f"User: {user_message}")

    # 3) Optionally append the image (if provided) 
    #    (Typically, you'd only do this once on the first user message)
    #    The calling code can decide whether to pass pil_image or not.
    if pil_image is not None:
        conversation_contents.append(pil_image)

    # 4) Call Gemini with all conversation text + optional image
    response = client.models.generate_content(
        model=model_name,
        contents=conversation_contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instructions,
            temperature=0.7,  # or your desired value
        )
    )

    # Return the text output
    return response.text
