import time
import io
import re
import json

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from requests_ai_standalone import request_gpt4o

# 1. bring a cup from the kitchen, put water and place on the meeting table - easy
# 2. put the pet bottle and the can in the garbage - easy - no mark
# 3. put the slippers in the proper shelf - middle
# 4. set up the meeting table for 2 people !!!
# 5. turn on the television and put the remote controller in the printer area - easy
# 6. find a book about AI and place it above the fridge - middle - mark
# 7. prepare a desk for a new user !!!
# 8. throw out the garbage and place new garbage bag in the bin - middle - mark
# 9. refill the hand soap in the kitchen - middle -  no mark
# 10. make coffee !!!

# test marking images to identify objects - more marks for more difficult tasks
# set the user goal first before starting the task - planner should ask questions to clarify the goal
# make the testing 

def extract_json_snippet(text: str) -> str:
    """Extract the first JSON array from the assistant’s reply."""
    m = re.search(r"```json\s*(\[\s*\{.*?\}\s*\])\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)
    return m.group(1) if m else "[]"


def strip_rap_label_and_json(full_text: str) -> str:
    """
    Remove all fenced code blocks (```...```), drop any 'RAP:' lines,
    and return only the narrative text.
    """
    without_fences = re.sub(r"```[\s\S]*?```", "", full_text)
    lines = [l for l in without_fences.splitlines() if not l.strip().startswith("RAP:")]
    return "\n".join(lines).strip()


def handle_user_turn(user_text: str | None, images: list[Image.Image] | None):
    """Echoes the turn, calls GPT-4o with any images, and updates RAP tables."""
    # Echo user input
    if user_text:
        st.chat_message("user").write(user_text)

    # Display images in a 5-column grid as thumbnails
    if images:
        msg = st.chat_message("user")
        # chunk images into rows of 5
        for i in range(0, len(images), 5):
            row_imgs = images[i:i+5]
            cols = msg.columns(5)
            for col, img in zip(cols, row_imgs):
                col.image(img, width=300)

    # Build conversation context
    ctx = []
    for it in st.session_state["iteration_data"]:
        ctx += [
            {"role": "user",      "content": it["User"]},
            {"role": "assistant", "content": it["assistant_full"]},
        ]

    # Call GPT-4o
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            start = time.time()
            reply, tokens = request_gpt4o(
                user_message=user_text or "",
                conversation_history=ctx,
                pil_image=images or []
            )
            latency = time.time() - start
            narrative = strip_rap_label_and_json(reply)
            st.write(narrative)

    # Parse and record RAP JSON
    snippet = extract_json_snippet(reply)
    try:
        rap_list = json.loads(snippet)
    except:
        rap_list = []
    df = pd.DataFrame(rap_list)
    st.session_state["rap_versions"].append(df)

    # Track changes
    if len(st.session_state["rap_versions"]) > 1:
        prev, cur = st.session_state["rap_versions"][-2:]
        old_jsons = set(prev.apply(lambda r: r.to_json(), axis=1))
        is_new = cur.apply(lambda r: r.to_json() not in old_jsons, axis=1)
        iteration = len(st.session_state["rap_versions"])
        for _, row in cur[is_new].iterrows():
            st.session_state["rap_changes"].append({"Iteration": iteration, **row.to_dict()})

    # Record iteration metadata
    st.session_state["iteration_data"].append({
        "Iteration":      len(st.session_state["rap_versions"]),
        "User":           user_text or "<image>",
        "Assistant":      narrative,
        "assistant_full": reply,
        "gpt_time":       round(latency, 3),
        "tokens":         tokens
    })


def main():
    st.set_page_config(layout="wide")
    st.title("RAP-Bench: Live Multimodal Chat")

    # Initialize RAP history state
    st.session_state.setdefault("iteration_data", [])
    st.session_state.setdefault("rap_versions", [])
    st.session_state.setdefault("rap_changes", [])

    left, right = st.columns([2, 3])

    # Chat input panel with unified send
    with left:
        st.subheader("Chat")
        with st.form(key="chat_form", clear_on_submit=True):
            user_text = st.text_input("Type a command...", key="msg_text")
            uploaded_files = st.file_uploader(
                "Upload image(s)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="pending_uploads"
            )
            send = st.form_submit_button("Send")

        if send:
            images: list[Image.Image] = []
            if uploaded_files:
                images = [Image.open(f) for f in uploaded_files]

            # handle and display the chat
            handle_user_turn(user_text or None, images)

            # clear upload slots
            st.session_state.pop("pending_uploads", None)

    # RAP display panel
    with right:
        st.subheader("Robot Action Plan")
        if st.session_state["rap_versions"]:
            latest = st.session_state["rap_versions"][-1]
            
            if len(st.session_state["rap_versions"]) > 1:
                prev = st.session_state["rap_versions"][-2]
                new_cols = [c for c in latest.columns if c not in prev.columns]
                prev_dicts = [r.to_dict() for _, r in prev.iterrows()]
        
                def is_identical(row, ref):
                    shared = set(row.index) & set(ref.index)
                    return all(row[c] == ref[c] for c in shared)
        
                def highlight(row):
                    if not any(is_identical(row, pd.Series(pr)) for pr in prev_dicts):
                        return ["color: yellow"] * len(row)
                    if any(pd.notnull(row[c]) and row[c] not in ("", None) for c in new_cols):
                        return ["color: yellow"] * len(row)
                    return [""] * len(row)
        
                st.dataframe(latest.style.apply(highlight, axis=1), use_container_width=True)
            else:
                st.dataframe(latest, use_container_width=True)
        else:
            st.info("Your RAP will appear here once you chat.")

        # Change Log
        with st.expander("Change Log", expanded=False):
            if st.session_state["rap_changes"]:
                st.table(pd.DataFrame(st.session_state["rap_changes"]))
            else:
                st.write("No changes yet.")

        # Iteration History
        with st.expander("Iteration History", expanded=False):
            if st.session_state["iteration_data"]:
                hist = pd.DataFrame(st.session_state["iteration_data"]).rename(
                    columns={"gpt_time": "GPT latency (s)", "tokens": "Tokens"}
                )[["Iteration", "User", "Assistant", "GPT latency (s)", "Tokens"]]
                st.table(hist)
            else:
                st.write("No iterations yet.")

    # Per-Iteration Metrics
    if st.session_state["rap_versions"]:
        st.write("---")
        st.subheader("Per-Iteration Metrics")
        iterations = list(range(1, len(st.session_state["rap_versions"]) + 1))
        metrics = {
            "# Actions": [len(df) for df in st.session_state["rap_versions"]],
            "# Questions": [d["assistant_full"].count("?") for d in st.session_state["iteration_data"]],
            "Avg. Resp. Length": [len(d["assistant_full"].split()) for d in st.session_state["iteration_data"]],
            "# RAP Columns": [len(df.columns) for df in st.session_state["rap_versions"]],
            "GPT latency (s)": [d["gpt_time"] for d in st.session_state["iteration_data"]],
            "Tokens / Iter": [d["tokens"] for d in st.session_state["iteration_data"]],
        }
        cols = st.columns(len(metrics))
        for (title, data), col in zip(metrics.items(), cols):
            fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
            ax.plot(iterations, data, marker="o", linewidth=2)
            ax.set_title(title, pad=8)
            ax.set_xlabel("Iter")
            ax.set_ylabel(title if title else "")
            ax.set_xticks(iterations)
            ax.set_ylim(0, max(data) * 1.2 if data else 1)
            ax.grid(alpha=0.3)
            col.pyplot(fig, use_container_width=True, clear_figure=True)

    # Export All Data to Excel
    st.write("---")
    if st.button("📥 Export all data to Excel"):
        rows = []
        for i, d in enumerate(st.session_state["iteration_data"], start=1):
            snippet = extract_json_snippet(d["assistant_full"])
            rap_text = snippet.replace("\n", " ").strip()
            changes = [e for e in st.session_state["rap_changes"] if e["Iteration"] == i]
            rows.append({
                "Iteration": i,
                "User message": d["User"],
                "Assistant narrative": d["Assistant"],
                "RAP JSON": rap_text,
                "Change log": "; ".join(str({k: v for k, v in e.items() if k != "Iteration"}) for e in changes),
                "# Actions": len(st.session_state["rap_versions"][i - 1]),
                "# Questions": d["assistant_full"].count("?"),
                "Avg resp. length": len(d["assistant_full"].split()),
                "# RAP columns": len(st.session_state["rap_versions"][i - 1].columns),
                "GPT latency (s)": d["gpt_time"],
                "Tokens": d["tokens"]
            })
        export_df = pd.DataFrame(rows)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            export_df.to_excel(writer, "RAP_Bench", index=False)
        buf.seek(0)
        st.download_button(
            "Download as Excel",
            data=buf,
            file_name="rap_bench_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
