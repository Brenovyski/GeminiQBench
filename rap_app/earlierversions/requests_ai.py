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
    draw = ImageDraw.Draw(img)

    base_colors = [
        'red','green','blue','yellow','orange','pink','purple','brown','gray',
        'beige','turquoise','cyan','magenta','lime','navy','maroon','teal',
        'olive','coral','lavender','violet','gold','silver'
    ]
    colors = base_colors + additional_colors

    bb_json = parse_json(bounding_boxes)
    try:
        boxes = json.loads(bb_json)
    except:
        boxes = []

    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except IOError:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        y1, x1, y2, x2 = box.get("box_2d", [0, 0, 0, 0])
        abs_y1 = int((y1 / 1000) * height)
        abs_x1 = int((x1 / 1000) * width)
        abs_y2 = int((y2 / 1000) * height)
        abs_x2 = int((x2 / 1000) * width)
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
            temperature=0,
            safety_settings=SAFETY_SETTINGS,
        )
    )
    raw = response.text
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
    pil_image=None,  # either a single Image or a list of Images
) -> dict:
    """
    Build a SoM‐style payload for GPT‐4o that handles single or multiple images.
    """
    system_instructions: str = (
        '''
        Summary: You are a Robot Task Planner. Every time the user speaks, you must generate or refine a Robot Action Plan (RAP).

        0. **Absolute MUST-DO commands**
            - **Always** generate a RAP in response to every user input (even if ambiguous). You always use the latest RAP as a base and refine it.
            - **Never** rewrite the entire RAP from scratch; always build on the previous one.
            - When moving between different images or views, always include explicit MOVE actions to navigate between those scene positions before object retrieval steps.

        1. **Response Structure**  
           - **First**, reply to the user with any observations, do not ask questions yet.  
           - **Then**, on a new line, output exactly:
             ```
             RAP:
             ```
             followed immediately by a JSON array of objects.

        2. **RAP JSON Format**  
           - The top-level JSON must be an array `[...]`.  
           - Each element must be an object with at least these keys:  
             `"ACTION"`, `"OBJECT"`, `"ROBOT_POSITION"`, `"GRIPPER_L"`, `"GRIPPER_R"`  
             (you may add additional fields like `"HEAT"`, `"DURATION"`, etc., as needed).  
           - Coordinates inside the plan must be expressed in easy-to-read form (e.g. `[x, y]`) matching the image’s pixel dimensions.
           - Always increment or slightly change the RAP; never rewrite it completely.

        3. **Object Coordinates**  
           - After the JSON array, provide a bullet list titled **“Easy-to-read object coordinates:”**  
           - List each main object you referenced in the RAP, with its approximate `[x, y]` position.

        3.1 **Multi-Image navigation**
            - If multiple images (e.g. different rooms/views) are provided, include explicit MOVE actions
              to navigate between those scene positions before object retrieval steps.
            - You can ask about rooms, views, or scene positions to clarify navigation needs.
            - Always ensure the RAP includes navigation steps when moving between different images or views.
              One important note is tat the view navigation instruction must be very explicit, you can add another column to the RAP JSON to identify the view or room, 
              and then use that column to generate the MOVE action. Try to always indicate where the robot is located. 

        4. **Iterative Refinement**  
           - On **every** user input—even if ambiguous—generate a RAP.  
           - On subsequent inputs, refine the **previous** RAP; do **not** throw it away.  
           - You may add or remove individual actions but never rewrite the entire plan from scratch.

        5. **Follow-Up Questions**  
           - Always end your narrative response with any questions needed to gather missing details.
           - You can ask more than one question at the same time or the same question multiple times, but do not ask the same question twice in a row.  
           - Be creative and ask questions that help you refine the RAP.
           - Do not embed questions inside the JSON or coordinate list—questions belong **before** the `RAP:` marker.

        6. **Termination**  
           - When the user indicates they are satisfied, begin your final reply with:
             ```
             Final answer:
             ```
             then repeat the latest RAP JSON (and coordinates), tell the use that the final RAP is generate and stop asking further questions.

        '''
    )

    # Build messages
    content = [{"type": "input_text", "text": system_instructions.strip()}]

    # History
    for turn in (conversation_history or []):
        role = "User" if turn["role"] == "user" else "Assistant"
        content.append({"type": "input_text", "text": f"{role}: {turn['content']}"})

    # New user
    content.append({"type": "input_text", "text": f"User: {user_message}"})

    # Attach images: either one or many
    if pil_image:
        if isinstance(pil_image, list):
            for img in pil_image:
                b64 = encode_image_from_pil(img)
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}"
                })
        else:
            b64 = encode_image_from_pil(pil_image)
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}"
            })

    return {"model": "chatgpt-4o-latest", "input": [{"role": "user", "content": content}]}

def request_gpt4o(
    user_message: str,
    conversation_history: list,
    pil_image=None  # either a single Image or a list of Images
) -> tuple[str, int]:
    """
    Send the constructed payload to GPT-4o, returning (reply_text, total_tokens).
    """
    payload = _prepare_gpt4o_payload(
        conversation_history=conversation_history,
        user_message=user_message,
        pil_image=pil_image
    )
    response = client.responses.create(
        model=payload["model"],
        input=payload["input"]
    )

    usage = response.usage
    if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
        total = usage.prompt_tokens + usage.completion_tokens
    else:
        total = getattr(usage, "total_tokens", 0) or getattr(usage, "token_count", 0)

    return response.output_text, total
