# rap_app.py

import time
import io
import re
import json

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

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
    Remove all fenced code blocks (```...```), drop any 'RAP:' lines,
    and return only the narrative text.
    """
    without_fences = re.sub(r"```[\s\S]*?```", "", full_text)
    lines = [l for l in without_fences.splitlines() if not l.strip().startswith("RAP:")]
    return "\n".join(lines).strip()

def main():
    st.set_page_config(layout="wide")
    st.title("RAP-Bench: Interactive Robot Action Plan Refinement")

    # ─── Initialize session state ─────────────────────────────────────
    st.session_state.setdefault("rap_versions", [])
    st.session_state.setdefault("rap_changes", [])
    st.session_state.setdefault("original_images", [])     # list of PIL originals
    st.session_state.setdefault("masked_image", None)       # single PIL mask (only when exactly one image)
    st.session_state.setdefault("iteration_data", [])
    st.session_state.setdefault("last_mask_time", None)
    st.session_state.setdefault("mask_prompt", "Detect all objects in the image.")
    st.session_state.setdefault("image_mode", "Original")   # "Original" or "Masked" (only used when single image)

    # ─── 1) Multi-Image Uploader ─────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload one or more images (jpg/png):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="When you upload more than one image, masking is skipped and all originals are used."
    )

    if not uploaded_files:
        # Reset all state if nothing uploaded
        st.session_state["rap_versions"].clear()
        st.session_state["rap_changes"].clear()
        st.session_state["original_images"].clear()
        st.session_state["masked_image"] = None
        st.session_state["iteration_data"].clear()
        st.session_state["last_mask_time"] = None
        st.session_state["mask_prompt"] = "Detect all objects in the image."
        st.session_state["image_mode"] = "Original"
        st.info("Please upload at least one image to begin.")
        return

    # Load all uploaded files into PIL.Images
    # Only re‐load if the count changed (to avoid resetting on every rerun)
    if len(st.session_state["original_images"]) != len(uploaded_files):
        st.session_state["original_images"] = [Image.open(f) for f in uploaded_files]
        # Clear previously generated mask if any
        st.session_state["masked_image"] = None
        st.session_state["last_mask_time"] = None
        st.session_state["image_mode"] = "Original"
        st.session_state["iteration_data"].clear()
        st.session_state["rap_versions"].clear()
        st.session_state["rap_changes"].clear()

    originals = st.session_state["original_images"]
    num_images = len(originals)

    # ─── 2) If exactly one image: show Mask Prompt + Generate Mask ─────────────
    if num_images == 1:
        single_orig = originals[0]

        st.subheader("Masking Instructions (single image)")
        prompt = st.text_area(
            "Enter masking prompt:",
            value=st.session_state["mask_prompt"],
            key="mask_prompt",
            height=80,
        )

        if st.button("Generate Mask"):
            with st.spinner("Generating mask…"):
                t0 = time.time()
                masked = generate_masked_image(single_orig, prompt=prompt)
                mask_t = time.time() - t0
                st.session_state["masked_image"] = masked
                st.session_state["last_mask_time"] = mask_t

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(single_orig, use_container_width=True)
        with col2:
            st.subheader("Masked Image (512px high)")
            if st.session_state["masked_image"] is not None:
                m = st.session_state["masked_image"]
                # Downscale if height > 512
                w, h = m.size
                if h > 512:
                    scale = 512 / h
                    m = m.resize((int(w * scale), 512), Image.Resampling.LANCZOS)
                st.image(m, use_container_width=True)
                st.caption(f"Mask latency: {st.session_state['last_mask_time']:.3f} s")
            else:
                st.info("Click **Generate Mask** to see the masked image.")

        # If no mask yet, stop. Chat only appears after a mask is generated.
        if st.session_state["masked_image"] is None:
            return

        st.write("---")

        # 2b) Image Mode Selector (locks once chat begins)
        locked = len(st.session_state["iteration_data"]) > 0
        st.subheader("Which Image to Send to GPT-4o")
        st.radio(
            "Use for all requests:",
            options=["Original", "Masked"],
            index=0 if st.session_state["image_mode"] == "Original" else 1,
            disabled=locked,
            key="image_mode",
        )

    # ─── 3) If multiple images: show side-by-side originals, skip masking controls ───
    else:
        st.warning(
            f"You have uploaded {num_images} images. Masking is disabled when multiple images are present.\n"
            "All original images will be sent to GPT-4o on each request."
        )
        cols = st.columns(num_images)
        for idx, img in enumerate(originals):
            with cols[idx]:
                st.subheader(f"Original #{idx+1}")
                st.image(img, use_container_width=True)
        st.write("---")

    # ─── 4) Chat & RAP ─────────────────────────────────────────────────────────────
    left, right = st.columns([2, 3])

    # 4a) CHAT PANEL
    with left:
        st.subheader("Chat")
        user_input = st.chat_input("Type something like 'Make egg scramble'")
        if user_input:
            # Echo user
            st.chat_message("user").write(user_input)

            # Build context history
            ctx = []
            for it in st.session_state["iteration_data"]:
                ctx.append({"role": "user",      "content": it["User"]})
                ctx.append({"role": "assistant", "content": it["assistant_full"]})

            # Call GPT-4o
            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("Thinking…"):
                    t0 = time.time()

                    if num_images == 1:
                        # Single‐image mode: pick original or masked based on choice
                        if st.session_state["image_mode"] == "Original":
                            img_to_send = originals[0]
                        else:
                            img_to_send = st.session_state["masked_image"]
                        full_reply, tokens = request_gpt4o(
                            user_message=user_input,
                            conversation_history=ctx,
                            pil_image=img_to_send
                        )
                    else:
                        # Multi-image mode: send all originals as a list
                        # (request_gpt4o should be adapted internally to accept a list of PILs)
                        full_reply, tokens = request_gpt4o(
                            user_message=user_input,
                            conversation_history=ctx,
                            pil_image=originals  # a list
                        )

                    gpt_latency = time.time() - t0

                narrative = strip_rap_label_and_json(full_reply)
                placeholder.write(narrative)

            # Parse JSON RAP
            snippet = extract_json_snippet(full_reply)
            try:
                rap_list = json.loads(snippet)
                df = pd.DataFrame(rap_list)
            except:
                df = pd.DataFrame()
            st.session_state["rap_versions"].append(df)

            # Track changes
            if len(st.session_state["rap_versions"]) > 1:
                prev, cur = st.session_state["rap_versions"][-2:]
                old_jsons = set(prev.apply(lambda r: r.to_json(), axis=1))
                is_new = cur.apply(lambda r: r.to_json() not in old_jsons, axis=1)
                itn = len(st.session_state["rap_versions"])
                for _, row in cur.loc[is_new].iterrows():
                    st.session_state["rap_changes"].append({"Iteration": itn, **row.to_dict()})

            # Record iteration data
            st.session_state["iteration_data"].append({
                "Iteration":      len(st.session_state["rap_versions"]),
                "User":           user_input,
                "Assistant":      narrative,
                "assistant_full": full_reply,
                "gpt_time":       round(gpt_latency, 3),
                "tokens":         tokens
            })

    # 4b) RAP TABLE + Logs + Iteration History
    with right:
        st.subheader("Robot Action Plan")
        if st.session_state["rap_versions"]:
            latest = st.session_state["rap_versions"][-1]
            if len(st.session_state["rap_versions"]) > 1:
                prev = st.session_state["rap_versions"][-2]
                old_jsons = set(prev.apply(lambda r: r.to_json(), axis=1))
                is_new = latest.apply(lambda r: r.to_json() not in old_jsons, axis=1)

                def highlight_row(row):
                    return ["color: yellow"] * len(row) if is_new[row.name] else [""] * len(row)

                st.dataframe(latest.style.apply(highlight_row, axis=1),
                             use_container_width=True)
            else:
                st.dataframe(latest, use_container_width=True)
        else:
            st.info("Your RAP will appear here once you chat.")

        with st.expander("Change Log", expanded=False):
            if st.session_state["rap_changes"]:
                st.table(pd.DataFrame(st.session_state["rap_changes"]))
            else:
                st.write("No changes yet.")

        with st.expander("Iteration History", expanded=False):
            if st.session_state["iteration_data"]:
                hist_df = pd.DataFrame(st.session_state["iteration_data"])[
                    ["Iteration", "User", "Assistant", "gpt_time", "tokens"]
                ]
                hist_df = hist_df.rename(columns={
                    "gpt_time": "GPT latency (s)",
                    "tokens":   "Tokens"
                })
                st.table(hist_df)
            else:
                st.write("No iterations yet.")

    # ─── 5) Per-Iteration Metrics ─────────────────────────────────────────────────
    if st.session_state["rap_versions"]:
        st.write("---")
        st.subheader("Per-Iteration Metrics")
        its  = list(range(1, len(st.session_state["rap_versions"]) + 1))
        acts = [len(df) for df in st.session_state["rap_versions"]]
        qs   = [d["assistant_full"].count("?") for d in st.session_state["iteration_data"]]
        arl  = [len(d["assistant_full"].split()) for d in st.session_state["iteration_data"]]
        cols = [len(df.columns) for df in st.session_state["rap_versions"]]
        gt   = [d["gpt_time"] for d in st.session_state["iteration_data"]]
        tc   = [d["tokens"] for d in st.session_state["iteration_data"]]

        specs = [
            ("# Actions",         acts, "# Actions"),
            ("# Questions",       qs,  "# Questions"),
            ("Avg. Resp. Length", arl, "Words"),
            ("# RAP Columns",     cols, "# Columns"),
            ("GPT latency (s)",   gt,   "Seconds"),
            ("Tokens / Iter",     tc,   "Tokens"),
        ]
        chart_cols = st.columns(len(specs))
        for (title, data, ylabel), col in zip(specs, chart_cols):
            fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
            ax.plot(its, data, marker="o", linewidth=2)
            ax.set_title(title, pad=8)
            ax.set_xlabel("Iter")
            ax.set_ylabel(ylabel)
            ax.set_xticks(its)
            ax.set_ylim(0, max(data) * 1.2 if data else 1)
            ax.grid(alpha=0.3)
            col.pyplot(fig, use_container_width=True, clear_figure=True)

    # ─── 6) Export All Data to Excel ───────────────────────────────────────────────
    st.write("---")
    if st.button("📥 Export all data to Excel"):
        rows = []
        for i, d in enumerate(st.session_state["iteration_data"], start=1):
            snippet = extract_json_snippet(d["assistant_full"])
            rap_str = snippet.replace("\n", " ").strip()
            changes = [
                {k: v for k, v in e.items() if k != "Iteration"}
                for e in st.session_state["rap_changes"] if e["Iteration"] == i
            ]
            rows.append({
                "Iteration":            i,
                "User message":         d["User"],
                "Assistant narrative":  d["Assistant"],
                "RAP JSON":             rap_str,
                "Change log":           "; ".join(map(str, changes)),
                "# Actions":            len(st.session_state["rap_versions"][i - 1]),
                "# Questions":          d["assistant_full"].count("?"),
                "Avg resp. length":     len(d["assistant_full"].split()),
                "# RAP columns":        len(st.session_state["rap_versions"][i - 1].columns),
                "GPT latency (s)":      d["gpt_time"],
                "Mask prompt":          (st.session_state["mask_prompt"] 
                                         if num_images == 1 else ""),
                "Masking latency (s)":  (f"{st.session_state['last_mask_time']:.3f}" 
                                         if (num_images == 1 and st.session_state["last_mask_time"] is not None) 
                                         else ""),
                "Tokens":               d["tokens"]
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