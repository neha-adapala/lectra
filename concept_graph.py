import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import fitz
import streamlit.components.v1 as comp
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile
import os
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

# ── Color scheme ──────────────────────────────────────────────────────────────
# Primary:    Deep Blue     #1B2A4A
# Background: Soft Off-White #F7F5F0
# Accent:     Teal          #2A9D8F
# Highlight:  Warm Yellow   #E9C46A

st.set_page_config(layout="wide", page_title="Lectra")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F7F5F0;
    color: #1B2A4A;
}

/* ── App background ── */
.stApp {
    background-color: #F7F5F0;
}

/* ── Main content area ── */
.block-container {
    margin={"top":50}
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 100%;
}

/* ── Page title ── */
h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 5rem !important;
    color: #1B2A4A !important;
    letter-spacing: -0.02em !important;
    margin-top: 10rem;
    text-align: center !important;

}

/* ── Section headings (###) ── */
h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #1B2A4A !important;
    margin-top: 3.5 rem !important;
    letter-spacing: -0.01em !important;
    text-align: center !important;

}

h3 {
    font-size: 1.1rem !important;
    margin-bottom: 0.75rem !important;
}

h1 a, h2 a, h3 a {
    display: none !important;
}

/* ── Graph title headings rendered via st.markdown ── */
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #1B2A4A !important;
}

/* ── Labels and text ── */
label, p, span, div {
    font-family: 'Inter', sans-serif;
    color: #1B2A4A;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.25rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #1B2A4A;
}
[data-testid="stFileUploader"] label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    color: #1B2A4A !important;
}

/* ── Text area ── */
textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important;
    color: #1B2A4A !important;
    background: #ffffff !important;
    border: 1.5px solid #d4cfc6 !important;
    border-radius: 8px !important;
    transition: border-color 0.2s !important;
}
textarea:focus {
    border-color: #2A9D8F !important;
    box-shadow: 0 0 0 3px rgba(42, 157, 143, 0.15) !important;
}

/* ── Radio buttons ── */
[data-testid="stRadio"] label {
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important;
    color: #1B2A4A !important;
}
[data-testid="stRadio"] > label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
}

/* ── Primary button ── */
[data-testid="stButton"] button[kind="primary"],
.stButton > button[kind="primary"] {
    background-color: #ffffff !important;
    color: #1b2a4a !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.01em;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.5rem !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #ffffff !important;
    transform: none !important;
}
[data-testid="stButton"] button[kind="primary"]:disabled {
    background-color: #b0bec5 !important;
    transform: none !important;
}

/* ── Secondary / default buttons ── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    border: 1.5px solid #1B2A4A !important;
    color: #1B2A4A !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background-color: #1B2A4A !important;
    color: #F7F5F0 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1.5px solid #e4e0d8;
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    color: #1B2A4A !important;
    padding: 0.75rem 1rem !important;
    background: #f0ede6;
}
[data-testid="stExpander"] summary:hover {
    background: #e8e4dc !important;
}

/* ── Info / warning boxes ── */
[data-testid="stInfo"] {
    background: rgba(42, 157, 143, 0.08) !important;
    border-left: 4px solid #2A9D8F !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    color: #1B2A4A !important;
}
[data-testid="stWarning"] {
    background: rgba(233, 196, 106, 0.15) !important;
    border-left: 4px solid #E9C46A !important;
    border-radius: 8px !important;
    color: #1B2A4A !important;
}

/* ── Columns divider ── */
[data-testid="column"]:first-child {
    border-right: 1px solid #e4e0d8;
    padding-right: 1.5rem;
}
[data-testid="column"]:last-child {
    padding-left: 1.5rem;
}

/* ── Sidebar (if used) ── */
[data-testid="stSidebar"] {
    background: #1B2A4A !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F7F5F0; }
::-webkit-scrollbar-thumb { background: #2A9D8F; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #1B2A4A; }

/* ── Graph section label ── */
.graph-label {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: #1B2A4A;
    padding: 0.5rem 0.9rem;
    background: #ffffff;
    border-left: 4px solid #E9C46A;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.75rem;
    display: inline-block;
}

/* ── Placeholder text ── */
::placeholder {
    color: #9e9b94 !important;
    font-family: 'Inter', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.title("lectra")
st.subheader("conceptually map your notes")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "annotations": [],
    "graphs": [],
    "pdf_text_cache": None,
    "selected_text": "",
    "selected_label": 3,
    "last_seen_by_level": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# loading the model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# pdf opening
def pdf_text(file):
    notes = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""
    for page in notes:
        full_text += page.get_text(sort=True)
    notes.close()
    return full_text

# rendering graph
def render_graph(graph_data, graph_index):
    root_label      = graph_data["root_label"]
    nodes           = graph_data["nodes"]
    node_embeddings = graph_data["node_embeddings"]

    # Dusky teal for H1 root titles
    # Light yellow for H2 subtitles
    # Navy for sentence/body nodes
    DUSKY_TEAL   = "#a2c7c1"   # H1 root — dusky, muted teal
    LIGHT_YELLOW = "#F5E6A3"   # H2 titles — soft warm yellow
    TEAL_LIGHT   = "#7ab5ae"   # H3 subtitles — lighter dusky teal
    NAVY         = "#1B2A4A"   # sentence nodes — deep navy

    color_map = {
        2: LIGHT_YELLOW,      # H2 — light yellow
        3: TEAL_LIGHT,        # H3 — lighter teal
        "sentence": NAVY,     # sentence — navy
    }
    size_map = {2: 52, 3: 40, "sentence": 22}

    net = Network(height="100%", width="100%", directed=False,
                  bgcolor="#F7F5F0", font_color="#1B2A4A")

    # Root node — dusky teal
    net.add_node(f"root_{graph_index}",
                 label=root_label,
                 color={
                     "background": DUSKY_TEAL,
                     "border": "#2e5e57",
                     "highlight": {"background": "#3a6e66", "border": "#F7F5F0"}
                 },
                 size=100,
                 title=root_label,
                 font={"size": 17, "color": "#F7F5F0", "bold": True},
                 shape="ellipse",
                 borderWidth=2,
                 borderWidthSelected=3,
                 margin={"top": 16, "right": 20, "bottom": 16, "left": 20})

    for node in nodes:
        level = node["level"]
        if level in (2, 3):
            display_label = node["text"]
            shape = "box"
            if level == 2:
                font_size = 14
                border_color = "#c8b84e"
                font_color = "#2a2200"   # dark text on light yellow
                bg = LIGHT_YELLOW
            else:
                font_size = 12
                border_color = "#3a7a72"
                font_color = "#F7F5F0"   # light text on teal
                bg = TEAL_LIGHT
        else:
            display_label = "·"
            shape = "dot"
            font_size = 9
            border_color = "#2A9D8F"
            font_color = "#F7F5F0"
            bg = NAVY

        net.add_node(node["node_id"],
                     label=display_label,
                     color={
                         "background": bg,
                         "border": border_color,
                         "highlight": {
                             "background": "#E9C46A",
                             "border": "#1B2A4A"
                         }
                     },
                     size=size_map.get(level, size_map["sentence"]),
                     title=node["text"],
                     font={"size": font_size, "color": font_color},
                     shape=shape,
                     margin={"top": 14, "right": 18, "bottom": 14, "left": 18})

    # Hierarchy / chain edges
    for node in nodes:
        parent = node.get("parent_node_id")
        src = parent if parent else f"root_{graph_index}"
        net.add_edge(src, node["node_id"], color="#2A9D8F", width=1.5)

    # Cosine similarity edges
    if len(node_embeddings) >= 2:
        emb_matrix = np.array(node_embeddings)
        sim_matrix = cosine_similarity(emb_matrix)
        n = len(nodes)
        hierarchy_pairs = set()
        for node in nodes:
            p = node.get("parent_node_id") or f"root_{graph_index}"
            hierarchy_pairs.add((p, node["node_id"]))
            hierarchy_pairs.add((node["node_id"], p))

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] > 0.6:
                    ni = nodes[i]["node_id"]
                    nj = nodes[j]["node_id"]
                    if (ni, nj) not in hierarchy_pairs:
                        net.add_edge(ni, nj,
                                     color="#E9C46A", width=1, dashes=True,
                                     title=f"similarity: {sim_matrix[i][j]:.2f}")

    net.set_options("""
    {
      "layout": {
        "improvedLayout": true
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -12000,
          "centralGravity": 0.15,
          "springLength": 250,
          "springConstant": 0.025,
          "damping": 0.1
        },
        "minVelocity": 0.75
      },
      "edges": { "smooth": { "type": "dynamic" } },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": true,
        "keyboard": true,
        "zoomView": true
      }
    }
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        tmp_path = f.name
    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        raw_html = f.read()
    os.unlink(tmp_path)

    uid    = f"wrap_{graph_index}"
    btn_id = f"fsbtn_{graph_index}"

    fullscreen_block = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=Inter:wght@400;500&display=swap');
    #{uid} {{
        position: relative;
        width: 100%;
        height: 900px;
        transition: all 0.25s ease;
        border-radius: 12px;
        overflow: hidden;
        border: 1.5px solid #e4e0d8;
    }}
    #{uid}.fullscreen {{
        position: fixed !important;
        top: 0; left: 0;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 99999;
        background: #F7F5F0;
        border-radius: 0;
    }}
    #{btn_id} {{
        position: absolute; top: 12px; right: 12px; z-index: 100000;
        background: #1B2A4A; color: #F7F5F0; border: none;
        padding: 6px 16px; border-radius: 6px; cursor: pointer;
        font-size: 13px; font-family: 'Space Grotesk', sans-serif; font-weight: 600;
        letter-spacing: 0.01em;
    }}
    #{btn_id}:hover {{ background: #2A9D8F; }}
    #mynetwork {{ width: 100% !important; height: 1000px !important; background: #F7F5F0 !important; }}
    </style>
    <div id="{uid}">
    """

    close_block = f"""
    </div>
    <script>
    function toggleFS_{graph_index}() {{
        var wrap = document.getElementById('{uid}');
        var btn  = document.getElementById('{btn_id}');
        wrap.classList.toggle('fullscreen');
        btn.textContent = wrap.classList.contains('fullscreen') ? 'Exit' : 'Expand';
        if (window.network) setTimeout(() => window.network.fit(), 300);
    }}
    </script>
    """

    html = raw_html.replace("<body>", "<body>" + fullscreen_block)
    html = html.replace("</body>", close_block + "</body>")
    return html

# file upload
uploaded_file = st.file_uploader("Upload PDF notes", type="pdf")

if not uploaded_file:
    st.session_state.pdf_text_cache = None
    st.session_state.selected_text = ""

if uploaded_file:
    if st.session_state.pdf_text_cache is None:
        st.session_state.pdf_text_cache = pdf_text(uploaded_file)

    text = st.session_state.pdf_text_cache
    col1, col2 = st.columns([2, 1])

    with col1:
        comp.html(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Inter:wght@300;400;500&display=swap');
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                background: #F7F5F0;
                font-family: 'Inter', sans-serif;
                color: #1B2A4A;
            }}
    
            ::selection {{
                background: rgba(233, 196, 106, 0.35);
            }}
        </style>
        <div id="pdfcontent">{text.replace(chr(10), '<br>')}</div>
        
        """, height=600, scrolling=True)

    with col2:
        st.markdown("### Annotate")
        edited_text = st.text_area(
            "Selected text",
            value=st.session_state.selected_text,
            height=180,
            placeholder="Copy text here"
        )
        label_names = {0: "H1", 1: "H2", 2: "H3", 3: "Text"}
        selected_label = st.radio(
            "Label",
            options=[0, 1, 2, 3],
            format_func=lambda x: label_names[x],
            index=st.session_state.selected_label,
            horizontal=True
        )

        if st.button("Submit", type="primary", disabled=not edited_text.strip()):
            clean  = edited_text.strip()
            ann_id = len(st.session_state.annotations)

            if selected_label == 0:
                st.session_state.graphs.append({
                    "root_label": clean,
                    "nodes": [],
                    "node_embeddings": [],
                })
                st.session_state.last_seen_by_level = {}
                st.session_state.annotations.append([ann_id, 0, None, clean])

            else:
                if not st.session_state.graphs:
                    st.warning("Please submit an H1 title first.")
                    st.stop()

                g = st.session_state.graphs[-1]

                parent_ann_id = None
                for lvl in range(selected_label - 1, 0, -1):
                    if lvl in st.session_state.last_seen_by_level:
                        parent_ann_id = st.session_state.last_seen_by_level[lvl]
                        break

                parent_node_id = None
                if parent_ann_id is not None:
                    for nd in g["nodes"]:
                        if nd["ann_id"] == parent_ann_id and nd["type"] == "heading":
                            parent_node_id = nd["node_id"]
                            break

                st.session_state.annotations.append([ann_id, selected_label, parent_ann_id, clean])

                if selected_label in (1, 2):
                    node_id = f"ann_{ann_id}_h"
                    g["nodes"].append({
                        "node_id": node_id,
                        "ann_id": ann_id,
                        "type": "heading",
                        "level": selected_label + 1,
                        "text": clean,
                        "parent_node_id": parent_node_id,
                    })
                    g["node_embeddings"].append(model.encode([clean])[0])

                else:
                    heading_parent = parent_node_id
                    if heading_parent is None:
                        for nd in reversed(g["nodes"]):
                            if nd["type"] == "heading":
                                heading_parent = nd["node_id"]
                                break

                    sentences = sent_tokenize(clean) or [clean]
                    prev = heading_parent
                    for s_idx, sentence in enumerate(sentences):
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        node_id = f"ann_{ann_id}_s{s_idx}"
                        g["nodes"].append({
                            "node_id": node_id,
                            "ann_id": ann_id,
                            "type": "sentence",
                            "level": "sentence",
                            "text": sentence,
                            "parent_node_id": prev,
                        })
                        g["node_embeddings"].append(model.encode([sentence])[0])
                        prev = node_id

                st.session_state.last_seen_by_level[selected_label] = ann_id
                for lvl in range(selected_label + 1, 4):
                    st.session_state.last_seen_by_level.pop(lvl, None)

            print(f"annotation {ann_id} level={selected_label} '{clean[:50]}'")
            st.session_state.selected_text = ""
            st.session_state.selected_label = 3
            st.rerun()
            
            

# making the graphs
if st.session_state.graphs:
    for g_idx, g_data in enumerate(st.session_state.graphs):
        st.markdown(
            f'<div class="graph-label">Conceptual Graph {g_idx + 1}: {g_data["root_label"]}</div>',
            unsafe_allow_html=True
        )
        if g_data["nodes"]:
            graph_html = render_graph(g_data, g_idx)
            comp.html(graph_html, height=10000, scrolling=False)
        else:
            st.info("Add subheadings to populate the graph.")
else:
    st.info("Upload a PDF to get started.")