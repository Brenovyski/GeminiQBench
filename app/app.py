import streamlit as st
from PIL import Image
import gemini  # <-- This is the gemini.py module you created

def main():
    st.set_page_config(layout="wide")
    st.title("GeminiQBench: Gemini 2.0 Object Detection + Multi-Turn Chat")

    # Session state to store conversation
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []

    # Step 1: Image Upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    masked_image = None

    if uploaded_file is not None:
        # Create two columns for displaying images
        col1, col2 = st.columns(2)

        # Show original image in the first column
        with col1:
            st.subheader("Original Image")
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_container_width=True)

        # Generate and show masked image in the second column
        with col2:
            st.subheader("Masked Image (Gemini 2.0 Object Detection)")
            with st.spinner("Processing image with Gemini 2.0..."):
                masked_image = gemini.generate_masked_image(
                    original_image,
                    prompt="Detect all objects in the image."
                )
            st.image(masked_image, caption="Detected Objects", use_container_width=True)

    st.write("---")
    st.subheader("Chat with Gemini 1.5 Flash")

    # Step 2: Chat interface
    user_input = st.text_input("Type your message here...")
    if st.button("Send"):
        if not uploaded_file or not masked_image:
            st.warning("Please upload an image first to generate the masked image.")
        else:
            # Add the user message to conversation history
            st.session_state["conversation_history"].append(
                {"role": "user", "content": user_input}
            )

            # Call Gemini 1.5 Flash with the masked image as attachment
            response_text = gemini.request_gemini_chat(
                model_name="gemini-1.5-flash",  # Adjust if your model name differs
                user_message=user_input,
                pil_image=masked_image,  # The masked image as attachment
                conversation_history=st.session_state["conversation_history"],
                system_instructions=(
                    "You are a helpful assistant. Ask clarifying questions "
                    "based on the image and the user's messages."
                )
            )

            # Add the assistant response to conversation history
            st.session_state["conversation_history"].append(
                {"role": "assistant", "content": response_text}
            )

    # Display the conversation
    for turn in st.session_state["conversation_history"]:
        if turn["role"] == "user":
            st.markdown(f"**User:** {turn['content']}")
        else:
            st.markdown(f"**Assistant:** {turn['content']}")

if __name__ == "__main__":
    main()
