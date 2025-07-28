# ===== requests_ai_standalone.py =====
import os
import json
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# ---- Configuration ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Helper: Encode PIL Image ----
def encode_image_from_pil(image: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded JPEG string."""
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---- Prepare GPT-4o Payload ----
def _prepare_gpt4o_payload(
    conversation_history: list,
    user_message: str,
    pil_image=None,  # either a single Image or a list of Images
) -> dict:
    """
    Build a SoM‐style payload for GPT‐4o that handles single or multiple images.
    """
    system_instructions = '''
    Summary: You are a Robot Task Planner. Whenever the user speaks or uploads an image, integrate that image into your scene understanding—ask follow-up questions about it, add MOVE actions as needed, and refine the RAP accordingly.

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
           - Use the original image size for coordinates, not the resized version shown in the UI.

        3.1 **Multi-Image navigation**
            - If multiple images (e.g. different rooms/views) are provided, include explicit MOVE actions
              to navigate between those scene positions before object retrieval steps.
            - You can ask about rooms, views, or scene positions to clarify navigation needs.
            - Always ensure the RAP includes navigation steps when moving between different images or views.
              One important note is tat the view navigation instruction must be very explicit, you can add another column to the RAP JSON to identify the view or room, 
              and then use that column to generate the MOVE action. Try to always indicate where the robot is located. 
            - If you have any doubts about some item or view that is not clear or not visible in the image, ask the user to clarify. You should ask the user 
              to send a new image or clarify the view as needed.

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
    # Build message content
    content = [{"type": "input_text", "text": system_instructions.strip()}]

    # Append history
    for turn in conversation_history or []:
        role = "User" if turn["role"] == "user" else "Assistant"
        content.append({"type": "input_text", "text": f"{role}: {turn['content']}"})

    # New user message
    content.append({"type": "input_text", "text": f"User: {user_message}"})

    # Attach images uniformly
    imgs = pil_image or []
    if not isinstance(imgs, list):
        imgs = [imgs]
    for img in imgs:
        b64 = encode_image_from_pil(img)
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64}"
        })

    return {"model": "chatgpt-4o-latest", "input": [{"role": "user", "content": content}]}

# ---- Send Request to GPT-4o ----
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
    try:
        response = client.responses.create(
            model=payload["model"],
            input=payload["input"]
        )
    except Exception as e:
        st.error(f"GPT-4o request failed: {e}")
        return "", 0

    usage = getattr(response, 'usage', {})
    if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
        total = usage.prompt_tokens + usage.completion_tokens
    else:
        total = getattr(usage, 'total_tokens', 0) or getattr(usage, 'token_count', 0)

    return getattr(response, 'output_text', ''), total
