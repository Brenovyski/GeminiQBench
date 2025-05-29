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
    m = re.search(r"```json\s*(\[\s*{.*?}\s*\])\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"(\[\s*{.*?}\s*\])", text, re.DOTALL)
    return m.group(1) if m else "[]"

def strip_rap_label_and_json(full_text: str) -> str:
    without = re.sub(r"```[\s\S]*?```", "", full_text)
    lines = [l for l in without.splitlines() if not l.strip().startswith("RAP:")]
    return "\n".join(lines).strip()

def main():
    st.set_page_config(layout="wide")
    st.title("RAP-Bench: Interactive Robot Action Plan Refinement")

    # ─── Initialize session state ─────────────────────────────────────
    st.session_state.setdefault("rap_versions", [])
    st.session_state.setdefault("rap_changes", [])
    st.session_state.setdefault("masked_image", None)
    st.session_state.setdefault("iteration_data", [])
    st.session_state.setdefault("last_mask_time", None)

    # ─── 1) Image Uploader ───────────────────────────────────────────
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if not uploaded:
        # if cleared, reset everything
        st.session_state["rap_versions"].clear()
        st.session_state["rap_changes"].clear()
        st.session_state["masked_image"] = None
        st.session_state["iteration_data"].clear()
        st.session_state["last_mask_time"] = None
        st.info("Please upload an image to begin.")
        return

    original = Image.open(uploaded)
    st.session_state["original_image"] = original

    # ─── 1b) Mask Prompt & Generate Button ───────────────────────────
    st.subheader("Masking Instructions")
    prompt = st.text_area(
        "Enter masking prompt:",
        value=st.session_state.get("mask_prompt", "Detect all objects in the image."),
        key="mask_prompt",
        height=80,
    )

    if st.button("Generate Mask"):
        used = st.session_state["mask_prompt"]
        with st.spinner("Generating mask…"):
            t0 = time.time()
            full_masked = generate_masked_image(original, prompt=used)
            st.session_state["last_mask_time"] = time.time() - t0
            # downscale to 512px height
            w, h = full_masked.size
            if h > 512:
                scale = 512 / h
                full_masked = full_masked.resize(
                    (int(w * scale), 512),
                    Image.Resampling.LANCZOS
                )
        st.session_state["masked_image"] = full_masked

    # ─── 1c) Show Original & Masked ──────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original, width=600)
    with col2:
        st.subheader("Masked Image (512px high)")
        if st.session_state["masked_image"] is not None:
            st.image(st.session_state["masked_image"], width=600)
            st.caption(f"Mask latency: {st.session_state['last_mask_time']:.3f}s")
        else:
            st.info("Click **Generate Mask** to see the masked image.")

    # if no mask yet, stop here
    if st.session_state["masked_image"] is None:
        return

    st.write("---")

    # ─── 1d) Image Mode Selector (locks once chat begins) ─────────────
    locked = len(st.session_state["iteration_data"]) > 0
    st.subheader("Which Image to Send to GPT-4o")
    # radio manages its own session_state["image_mode"]
    st.radio(
        "Use for all requests:",
        options=["Original", "Masked"],
        index=0 if st.session_state.get("image_mode","Masked")=="Original" else 1,
        disabled=locked,
        key="image_mode",
    )

    st.write("---")

    # ─── 2) Chat & RAP ───────────────────────────────────────────────
    left, right = st.columns([2,3])

    # 2a) Chat Panel
    with left:
        st.subheader("Chat")
        user_input = st.chat_input("Type something like 'Make egg scramble'")
        if user_input:
            # echo
            st.chat_message("user").write(user_input)

            # build context history
            ctx = []
            for it in st.session_state["iteration_data"]:
                ctx.append({"role":"user","content":it["User"]})
                ctx.append({"role":"assistant","content":it["assistant_full"]})

            # GPT call
            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("Thinking…"):
                    t0 = time.time()
                    img_to_send = (original
                                   if st.session_state["image_mode"]=="Original"
                                   else st.session_state["masked_image"])
                    full_reply, tokens = request_gpt4o(
                        user_message=user_input,
                        conversation_history=ctx,
                        pil_image=img_to_send
                    )
                    gpt_latency = time.time() - t0

                narrative = strip_rap_label_and_json(full_reply)
                placeholder.write(narrative)

            # parse JSON RAP
            snippet = extract_json_snippet(full_reply)
            try:
                rap_list = json.loads(snippet)
                df = pd.DataFrame(rap_list)
            except:
                df = pd.DataFrame()
            st.session_state["rap_versions"].append(df)

            # track changes
            if len(st.session_state["rap_versions"])>1:
                prev, cur = st.session_state["rap_versions"][-2:]
                new = cur.apply(lambda r:r.to_json(),axis=1)
                old = prev.apply(lambda r:r.to_json(),axis=1)
                added = cur.loc[~new.isin(old)]
                itn = len(st.session_state["rap_versions"])
                for _,row in added.iterrows():
                    st.session_state["rap_changes"].append(
                        {"Iteration":itn, **row.to_dict()}
                    )

            # record iteration
            st.session_state["iteration_data"].append({
                "Iteration":      len(st.session_state["rap_versions"]),
                "User":           user_input,
                "Assistant":      narrative,
                "assistant_full": full_reply,
                "gpt_time":       round(gpt_latency,3),
                "tokens":         tokens
            })

    # 2b) RAP, Logs & History
    with right:
        st.subheader("Robot Action Plan")
        if st.session_state["rap_versions"]:
            latest = st.session_state["rap_versions"][-1]
            if len(st.session_state["rap_versions"]) > 1:
                prev = st.session_state["rap_versions"][-2]
                # Build a set of all previous rows' JSON strings
                old_jsons = set(prev.apply(lambda r: r.to_json(), axis=1))
        
                # Determine which rows in 'latest' are new
                is_new_row = latest.apply(lambda r: r.to_json() not in old_jsons, axis=1)
        
                # Define a function that highlights only truly new rows
                def highlight_text(row):
                    if is_new_row.loc[row.name]:
                        return ["color: yellow"] * len(row)
                    else:
                        return [""] * len(row)
        
                st.dataframe(
                    latest.style.apply(highlight_text, axis=1),
                    use_container_width=True
                )
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
                hist = pd.DataFrame(st.session_state["iteration_data"])[
                    ["Iteration","User","Assistant","gpt_time","tokens"]
                ]
                hist = hist.rename(columns={
                    "gpt_time":"GPT latency (s)",
                    "tokens":"Tokens"
                })
                st.table(hist)
            else:
                st.write("No iterations yet.")

    # ─── 3) Metrics ────────────────────────────────────────────────────
    if st.session_state["rap_versions"]:
        st.write("---")
        st.subheader("Per-Iteration Metrics")
        its  = list(range(1, len(st.session_state["rap_versions"])+1))
        acts = [len(df) for df in st.session_state["rap_versions"]]
        qs   = [d["assistant_full"].count("?") for d in st.session_state["iteration_data"]]
        arl  = [len(d["assistant_full"].split()) for d in st.session_state["iteration_data"]]
        cols = [len(df.columns) for df in st.session_state["rap_versions"]]
        gt   = [d["gpt_time"] for d in st.session_state["iteration_data"]]
        tc   = [d["tokens"] for d in st.session_state["iteration_data"]]

        specs = [
            ("# Actions", acts, "# Actions"),
            ("# Questions", qs, "# Questions"),
            ("Avg. Resp. Length", arl, "Words"),
            ("# RAP Columns", cols, "# Columns"),
            ("GPT latency (s)", gt, "Seconds"),
            ("Tokens / Iter", tc, "Tokens"),
        ]
        cols_ = st.columns(len(specs))
        for (title,data,ylabel),col in zip(specs,cols_):
            fig, ax = plt.subplots(figsize=(3,2),dpi=100)
            ax.plot(its, data, marker="o", linewidth=2)
            ax.set_title(title, pad=8)
            ax.set_xlabel("Iter"); ax.set_ylabel(ylabel)
            ax.set_xticks(its)
            ax.set_ylim(0, max(data)*1.2 if data else 1)
            ax.grid(alpha=0.3)
            col.pyplot(fig, use_container_width=True, clear_figure=True)

    # ─── 4) Export ───────────────────────────────────────────────────────
    st.write("---")
    if st.button("📥 Export all data to Excel"):
        rows = []
        for i,d in enumerate(st.session_state["iteration_data"], start=1):
            snippet = extract_json_snippet(d["assistant_full"])
            rap_str = snippet.replace("\n"," ").strip()
            changes = [
                {k:v for k,v in e.items() if k!="Iteration"}
                for e in st.session_state["rap_changes"] if e["Iteration"]==i
            ]
            rows.append({
                "Iteration":            i,
                "User message":         d["User"],
                "Assistant narrative":  d["Assistant"],
                "RAP JSON":             rap_str,
                "Change log":           "; ".join(map(str,changes)),
                "# Actions":            len(st.session_state["rap_versions"][i-1]),
                "# Questions":          d["assistant_full"].count("?"),
                "Avg resp. length":     len(d["assistant_full"].split()),
                "# RAP columns":        len(st.session_state["rap_versions"][i-1].columns),
                "GPT latency (s)":      d["gpt_time"],
                "Masking latency (s)":  st.session_state["last_mask_time"],
                "Mask prompt":          st.session_state.get("mask_prompt", ""),
                "Tokens":               d["tokens"]
            })
        export_df = pd.DataFrame(rows)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            export_df.to_excel(w, "RAP_Bench", index=False)
        buf.seek(0)
        st.download_button(
            "Download as Excel",
            data=buf,
            file_name="rap_bench_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__=="__main__":
    main()
