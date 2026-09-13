from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import the compiled LangGraph app + helpers from the backend
# -----------------------------
from bwa_backend import app, slugify, IMAGE_BACKENDS

IMAGES_DIR = Path("images")


# -----------------------------
# Helpers
# -----------------------------
def bundle_zip(md_text: str, md_filename: str, image_paths: List[Path]) -> bytes:
    """MD + only the images this post actually references."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        for p in image_paths:
            if p.is_file():
                z.write(p, arcname=f"images/{p.name}")
    return buf.getvalue()


def images_zip(image_paths: List[Path]) -> Optional[bytes]:
    files = [p for p in image_paths if p.is_file()]
    if not files:
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=f"images/{p.name}")
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress if available; else invoke.
    Yields ("updates", {node: delta}) during the run and ("final", state) at the end.

    Note: the graph is run EXACTLY ONCE. "values" mode emits the full state after
    each superstep, so the last one is the final state — re-invoking afterwards
    would pay for the whole blog a second time.
    """
    try:
        last_values: Optional[Dict[str, Any]] = None
        for mode, chunk in graph_app.stream(inputs, stream_mode=["updates", "values"]):
            if mode == "values":
                last_values = chunk
            else:
                yield ("updates", chunk)
        if last_values is not None:
            yield ("final", last_values)
            return
    except Exception as e:  # older langgraph, or multi-mode unsupported
        st.caption(f"Streaming unavailable ({type(e).__name__}); falling back to a single invoke.")

    yield ("final", graph_app.invoke(inputs))


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    """Merge an 'updates' chunk ({node_name: delta}) into a running state snapshot."""
    if isinstance(step_payload, dict):
        for value in step_payload.values():
            if isinstance(value, dict):
                # `sections` accumulates across parallel workers instead of overwriting
                secs = value.get("sections")
                if isinstance(secs, list):
                    current_state["sections"] = (current_state.get("sections") or []) + secs
                    value = {k: v for k, v in value.items() if k != "sections"}
                current_state.update(value)
    return current_state


def plan_to_dict(plan_obj: Any) -> Optional[dict]:
    if plan_obj is None:
        return None
    if hasattr(plan_obj, "model_dump"):
        return plan_obj.model_dump()
    if isinstance(plan_obj, dict):
        return plan_obj
    return json.loads(json.dumps(plan_obj, default=str))


def blog_title_of(out: Dict[str, Any], final_md: str) -> str:
    plan = plan_to_dict(out.get("plan"))
    if plan and plan.get("blog_title"):
        return plan["blog_title"]
    return extract_title_from_md(final_md, "blog")


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    return Path(src.strip().lstrip("./")).resolve()


def referenced_images(md: str) -> List[Path]:
    """Local image files this markdown links to, in order, de-duplicated."""
    out: List[Path] = []
    for m in _MD_IMG_RE.finditer(md):
        src = (m.group("src") or "").strip()
        if src.startswith(("http://", "https://")):
            continue
        p = _resolve_image_path(src)
        if p not in out:
            out.append(p)
    return out


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith(("http://", "https://")):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    """.md files in the working directory, newest first (README excluded)."""
    files = [p for p in Path(".").glob("*.md") if p.is_file() and p.name.lower() != "readme.md"]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

st.title("Blog Writing Agent")

with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area("Topic", height=120)
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")
    st.caption(f"Image backends: {' → '.join(IMAGE_BACKENDS)}")

    st.divider()
    st.subheader("Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                title = extract_title_from_md(read_md_file(p), p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog") and selected_md_file:
            md_text = read_md_file(selected_md_file)
            # Loaded from disk: only the markdown survives; plan/evidence aren't persisted.
            st.session_state["last_out"] = {
                "plan": None,
                "evidence": [],
                "image_specs": [],
                "final": md_text,
            }
            st.session_state["loaded_from"] = selected_md_file.name

# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Layout
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    st.session_state.pop("loaded_from", None)
    status = st.status("Running graph…", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    try:
        for kind, payload in try_stream(app, inputs):
            if kind == "updates":
                if isinstance(payload, dict):
                    for node_name in payload:
                        if node_name != last_node:
                            status.write(f"➡️ Node: `{node_name}`")
                            last_node = node_name

                current_state = extract_latest_state(current_state, payload)

                plan_dict = plan_to_dict(current_state.get("plan")) or {}
                queries = current_state.get("queries") or []
                progress_area.json(
                    {
                        "mode": current_state.get("mode"),
                        "needs_research": current_state.get("needs_research"),
                        "queries": queries[:5] if isinstance(queries, list) else [],
                        "evidence_count": len(current_state.get("evidence") or []),
                        "tasks": len(plan_dict.get("tasks", [])) or None,
                        "images": len(current_state.get("image_specs") or []),
                        "sections_done": len(current_state.get("sections") or []),
                    }
                )
                log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

            elif kind == "final":
                st.session_state["last_out"] = payload
                status.update(label="✅ Done", state="complete", expanded=False)
                log("[final] received final state")
    except Exception as e:
        status.update(label="❌ Run failed", state="error", expanded=True)
        msg = str(e)
        if "free-models-per-day" in msg or "RateLimitError" in type(e).__name__:
            st.error(
                "OpenRouter free-tier daily limit hit (50 requests/day). "
                "Wait for the UTC-midnight reset, add credits, or set a paid "
                "`OPENROUTER_MODEL` in `.env`."
            )
        else:
            st.error(f"{type(e).__name__}: {msg[:1500]}")
        log(f"[error] {type(e).__name__}: {msg[:2000]}")

# Render last result (if any)
out = st.session_state.get("last_out")
if out:
    final_md = out.get("final") or ""
    loaded_from = st.session_state.get("loaded_from")
    if loaded_from:
        st.caption(f"Loaded from disk: `{loaded_from}` — plan and evidence aren't saved in the .md file.")

    # --- Plan tab ---
    with tab_plan:
        st.subheader("Plan")
        plan_dict = plan_to_dict(out.get("plan"))
        if not plan_dict:
            st.info("No plan found in output.")
        else:
            st.write("**Title:**", plan_dict.get("blog_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))
            if plan_dict.get("constraints"):
                st.write("**Constraints:** " + ", ".join(plan_dict["constraints"]))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citation": t.get("requires_citation"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (closed_book mode, or no Tavily key/results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Preview tab ---
    with tab_preview:
        st.subheader("Markdown Preview")
        if not final_md:
            st.warning("No final markdown found.")
        else:
            if "[IMAGE GENERATION FAILED]" in final_md:
                st.warning(
                    "One or more images failed to generate — the post keeps a blockquote "
                    "with the prompt and the backend error in their place."
                )

            render_markdown_with_local_images(final_md)

            blog_title = blog_title_of(out, final_md)
            md_filename = f"{slugify(blog_title)}.md"
            refs = referenced_images(final_md)

            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )
            st.download_button(
                "📦 Download Bundle (MD + images)",
                data=bundle_zip(final_md, md_filename, refs),
                file_name=f"{slugify(blog_title)}_bundle.zip",
                mime="application/zip",
            )

    # --- Images tab ---
    with tab_images:
        st.subheader("Images")
        specs = out.get("image_specs") or []
        refs = referenced_images(final_md)

        if not specs and not refs:
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.write("**Image plan:**")
                for spec in specs:
                    if hasattr(spec, "model_dump"):
                        spec = spec.model_dump()
                    with st.expander(f"{spec.get('placeholder')} — {spec.get('filename')}"):
                        st.write("**Alt:** " + str(spec.get("alt", "")))
                        st.write("**Caption:** " + str(spec.get("caption", "")))
                        if spec.get("mermaid"):
                            st.caption("Mermaid source (rendered by Kroki):")
                            st.code(spec["mermaid"], language="text")
                        st.caption("Image-model prompt (Gemini fallback):")
                        st.code(spec.get("prompt", ""), language="text")

            missing = [p for p in refs if not p.is_file()]
            present = [p for p in refs if p.is_file()]

            if present:
                st.write("**Generated images:**")
                for p in present:
                    st.image(str(p), caption=p.name, use_container_width=True)
                z = images_zip(present)
                if z:
                    st.download_button(
                        "⬇️ Download Images (zip)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                    )
            elif refs:
                st.warning("This post links images that aren't on disk.")

            for p in missing:
                st.warning(f"Missing file: `images/{p.name}`")

            with st.expander("All files in images/"):
                all_files = sorted(p for p in IMAGES_DIR.iterdir() if p.is_file()) if IMAGES_DIR.exists() else []
                st.write(
                    "\n".join(f"- `{p.name}`" for p in all_files)
                    if all_files
                    else "_images/ is empty or missing._"
                )

    # --- Logs tab ---
    with tab_logs:
        st.subheader("Logs")
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
else:
    st.info("Enter a topic and click **Generate Blog**.")
