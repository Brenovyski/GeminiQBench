# ===== requests.py =====
import os
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageColor
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# ---- Configuration ----
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in environment variables.")

genai_client = genai.Client(api_key=GOOGLE_API_KEY)

# System instructions for bounding-box detection
BOUNDING_BOX_SYSTEM_INSTRUCTIONS = (
    "Return bounding boxes as a JSON array with labels. Never return masks or code fencing. "
    "Limit to 25 objects. If an object appears multiple times, name them by unique characteristics."
)
SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    )
]

# ---- Helpers from Spatial Understanding notebook ----

def parse_json(json_output: str) -> str:
    """
    Strip away markdown fences from the model’s JSON response.
    """
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```", 1)[0]
            break
    return json_output

# Build extended color palette
additional_colors = list(ImageColor.colormap.keys())

# ---- Generate Masked Image ----

def plot_bounding_boxes(im, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        im: PIL.Image to draw on.
        bounding_boxes: JSON string of bounding boxes.
    """
    img = im
    width, height = img.size
    # Create drawing object
    draw = ImageDraw.Draw(img)

    # Color list
    base_colors = [
        'red','green','blue','yellow','orange','pink','purple','brown','gray',
        'beige','turquoise','cyan','magenta','lime','navy','maroon','teal',
        'olive','coral','lavender','violet','gold','silver'
    ]
    colors = base_colors + additional_colors

    # Strip fences
    bb_json = parse_json(bounding_boxes)

    # Load font
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except IOError:
        font = ImageFont.load_default()

    # Draw each box
    for i, box in enumerate(json.loads(bb_json)):
        color = colors[i % len(colors)]
        y1, x1, y2, x2 = box.get("box_2d", [0,0,0,0])
        abs_y1 = int((y1/1000)*height)
        abs_x1 = int((x1/1000)*width)
        abs_y2 = int((y2/1000)*height)
        abs_x2 = int((x2/1000)*width)
        if abs_x1 > abs_x2: abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2: abs_y1, abs_y2 = abs_y2, abs_y1
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
        if "label" in box:
            draw.text((abs_x1+8, abs_y1+6), box["label"], fill=color, font=font)

    return img


def generate_masked_image(pil_image: Image.Image, prompt: str = "Detect all objects in the image.") -> Image.Image:
    """
    Use Gemini 2.0 Flash to detect objects, then draw bounding boxes using plot_bounding_boxes.
    """
    response = genai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, pil_image],
        config=types.GenerateContentConfig(
            system_instruction=BOUNDING_BOX_SYSTEM_INSTRUCTIONS,
            temperature=0.5,
            safety_settings=SAFETY_SETTINGS,
        )
    )
    raw = response.text
    # Draw boxes
    out_img = pil_image.copy()
    plot_bounding_boxes(out_img, raw)
    return out_img


# ---- GPT-4o Chat Setup ----
client = OpenAI()

def encode_image_from_pil(image: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded JPEG string."""
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _prepare_gpt4o_payload(
    conversation_history: list,
    user_message: str,
    pil_image: Image.Image = None,
    system_instructions: str = (
        "SUMMARY: You are a robot task planner. Generate or refine a Robot Action Plan (RAP). The RAP should be displayed after 'RAP: ' answer in the chat "
        "Each RAP must be output as a JSON array of objects with columns ACTION, OBJECT, ROBOT_POSITION, GRIPPER_L, GRIPPER_R, etc. "
        "After the answer include easy-to-read object coordinates. Generate the coordinates in the format [x, y] based on the image size."
        "Keep asking for more details until the user is satisfied. Keep asking questions to refine the RAP. After the user is satisfied, insert the answer with Final answer prefix."
        "About the RAP generation, every input of the user should trigger a new RAP generation. So the first command even though it is an ambiguous command, should also generate a RAP."
        "The next command should be used to refine the previous RAP. Important thing to notice is that you should maintain a conversation still and ask questions to the user to refine the RAP."
        "FORMAT OF THE OUTPUT: First part should be the respone to the user, and the second part should be the RAP. The RAP should be in JSON format. Separate these two part clearly with this text: 'RAP:' always in every reply"
    )
) -> dict:
    content = []
    # system
    content.append({"type":"input_text","text":system_instructions.strip()})
    # history
    for turn in (conversation_history or []):
        label = "User" if turn['role']=='user' else "Assistant"
        content.append({"type":"input_text","text":f"{label}: {turn['content']}"})
    # new user
    content.append({"type":"input_text","text":f"User: {user_message}"})
    # image
    if pil_image:
        b64 = encode_image_from_pil(pil_image)
        content.append({"type":"input_image","image_url":f"data:image/jpeg;base64,{b64}"})
    return {"model":"chatgpt-4o-latest","input":[{"role":"user","content":content}]}


def request_gpt4o(
    user_message: str,
    conversation_history: list,
    pil_image: Image.Image = None
) -> str:
    payload = _prepare_gpt4o_payload(
        conversation_history=conversation_history,
        user_message=user_message,
        pil_image=pil_image
    )
    response = client.responses.create(
        model=payload['model'],
        input=payload['input']
    )
    print(f"Response: {response.output_text}")
    return response.output_text