import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import json
import re

from requests_ai import generate_masked_image, request_gpt4o

def extract_json_snippet(text: str) -> str:
    """Extract the first JSON array from the assistant’s reply."""
    m = re.search(r"```json\s*(\[\s*{.*?}\s*\])\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"(\[\s*{.*?}\s*\])", text, re.DOTALL)
    return m.group(1) if m else "[]"

def strip_rap_label_and_json(full_text: str) -> str:
    """
    Remove any 'RAP:' label lines and strip out the JSON block,
    leaving only the narrative.
    """
    snippet = extract_json_snippet(full_text)
    narrative = full_text.replace(snippet, "").strip()
    lines = [l for l in narrative.splitlines() if not l.strip().startswith("RAP:")]
    return "\n".join(lines).strip()

def main():
    st.set_page_config(layout="wide")
    st.title("RAP-Bench: Interactive Robot Action Plan Refinement")

    # ─── Session State ──────────────────────────────────────────
    st.session_state.setdefault("rap_versions", [])
    st.session_state.setdefault("rap_changes", [])
    st.session_state.setdefault("masked_image", None)
    st.session_state.setdefault("iteration_data", [])

    # ─── 1) Image Uploader & Masking ────────────────────────────
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if not uploaded:
        # Reset if cleared
        st.session_state["rap_versions"].clear()
        st.session_state["rap_changes"].clear()
        st.session_state["masked_image"] = None
        st.session_state["iteration_data"].clear()
        st.info("Please upload an image to begin.")
        return

    original = Image.open(uploaded)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original, width=200)
    with col2:
        st.subheader("Masked Image (512px high)")
        if st.session_state["masked_image"] is None:
            with st.spinner("Running Gemini 2.0…"):
                full_masked = generate_masked_image(
                    original, prompt="Detect all objects in the image."
                )
                w, h = full_masked.size
                if h > 512:
                    scale = 512 / h
                    full_masked = full_masked.resize(
                        (int(w * scale), 512),
                        Image.Resampling.LANCZOS
                    )
            st.session_state["masked_image"] = full_masked
        st.image(st.session_state["masked_image"], width=200)

    st.write("---")

    # ─── 2) Chat & RAP Columns ──────────────────────────────────
    left, right = st.columns([2,3])

    # 2a) CHAT PANEL
    with left:
        st.subheader("Chat")
        user_input = st.chat_input("Type something like 'Make egg scramble'")

        if user_input:
            # Echo user message immediately
            st.chat_message("user").write(user_input)

            # Build context from past iterations
            context = []
            for entry in st.session_state["iteration_data"]:
                context.append({"role": "user",      "content": entry["User"]})
                context.append({"role": "assistant", "content": entry["assistant_full"]})

            # Placeholder + spinner for assistant
            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("Thinking…"):
                    full_reply = request_gpt4o(
                        user_message=user_input,
                        conversation_history=context,
                        pil_image=st.session_state["masked_image"]
                    )
                narrative = strip_rap_label_and_json(full_reply)
                placeholder.write(narrative)

            # Parse and store new RAP table
            snippet = extract_json_snippet(full_reply)
            try:
                rap_list = json.loads(snippet)
                df = pd.DataFrame(rap_list)
            except:
                df = pd.DataFrame()
            st.session_state["rap_versions"].append(df)

            # Compute and store change log entries
            if len(st.session_state["rap_versions"]) > 1:
                prev = st.session_state["rap_versions"][-2]
                cur  = st.session_state["rap_versions"][-1]
                new_j = cur.apply(lambda r: r.to_json(), axis=1)
                old_j = prev.apply(lambda r: r.to_json(), axis=1)
                added = cur.loc[~new_j.isin(old_j)]
                iteration = len(st.session_state["rap_versions"])
                for _, row in added.iterrows():
                    st.session_state["rap_changes"].append(
                        {"Iteration": iteration, **row.to_dict()}
                    )

            # Record this iteration’s data
            st.session_state["iteration_data"].append({
                "Iteration": len(st.session_state["rap_versions"]),
                "User": user_input,
                "Assistant": narrative,
                "assistant_full": full_reply
            })

    # 2b) RAP, Change Log & Iteration History
    with right:
        st.subheader("Robot Action Plan")

        # RAP Table
        if st.session_state["rap_versions"]:
            latest = st.session_state["rap_versions"][-1]
            if len(st.session_state["rap_versions"]) > 1:
                prev = st.session_state["rap_versions"][-2]
                nj = latest.apply(lambda r: r.to_json(), axis=1)
                oj = prev.apply(lambda r: r.to_json(), axis=1)
                mask = ~nj.isin(oj)
                def highlight_text(row):
                    # If this row is “new”, color all its text yellow; otherwise leave default
                    return ["color: yellow"] * len(row) if mask.loc[row.name] else [""] * len(row)
                st.dataframe(latest.style.apply(highlight_text, axis=1), use_container_width=True)
            else:
                st.dataframe(latest, use_container_width=True)
        else:
            st.info("Your RAP will appear here after the first response.")

        # Change Log expander
        with st.expander("Change Log", expanded=False):
            if st.session_state["rap_changes"]:
                st.table(pd.DataFrame(st.session_state["rap_changes"]))
            else:
                st.write("No changes logged yet.")

        # Iteration History expander
        with st.expander("Iteration History", expanded=False):
            if st.session_state["iteration_data"]:
                hist_df = pd.DataFrame(st.session_state["iteration_data"])[
                    ["Iteration", "User", "Assistant"]
                ]
                st.table(hist_df)
            else:
                st.write("No iterations yet.")

   # ─── 3) Compact Metrics ─────────────────────────────────────

    if st.session_state["rap_versions"]:
        # Re-use the same left/right pair from above, or
        # create a fresh one if you prefer:
        _, right_col = st.columns([2, 3])

        # Compute your per‐iteration counts
        counts = [len(df) for df in st.session_state["rap_versions"]]

        with right_col:
            st.write("---")
            st.subheader("Metrics: # Actions Over Iterations")
            # Make a small figure
            fig, ax = plt.subplots(figsize=(3, 2))   # 3″×2″
            ax.plot(counts, marker="o", linewidth=1)
            ax.set_xlabel("Iter")
            ax.set_ylabel("#Actions")
            ax.grid(True, alpha=0.3)
            # Show it
            st.pyplot(fig, clear_figure=True, use_container_width=False)

if __name__ == "__main__":
    main()
