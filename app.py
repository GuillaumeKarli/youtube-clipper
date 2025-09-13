import streamlit as st
import tempfile
import subprocess
from pathlib import Path
import re

st.set_page_config(page_title="YouTube Clip Extractor", page_icon="✂️", layout="centered")

st.title("✂️ YouTube Clip Extractor")
st.caption("Colle une URL YouTube + timecodes de début/fin → télécharge ton extrait (MP4).")

# --- fonction pour parser les timecodes ---
def parse_timecode(tc: str) -> float:
    tc = tc.strip().lower()
    h = m = s = 0.0
    if any(x in tc for x in ("h", "m", "s")):
        m1 = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?", tc)
        if not m1:
            raise ValueError(f"Timecode invalide: {tc}")
        h = float(m1.group(1) or 0)
        m = float(m1.group(2) or 0)
        s = float(m1.group(3) or 0)
        return h*3600 + m*60 + s
    if ":" in tc:
        parts = [float(p) for p in tc.split(":")]
        if len(parts) == 2:
            m, s = parts
            return m*60 + s
        elif len(parts) == 3:
            h, m, s = parts
            return h*3600 + m*60 + s
    return float(tc)

# --- fonction générique pour exécuter une commande ---
def run(cmd: list):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# --- télécharger la vidéo ---
def download_best(url: str, outdir: Path) -> Path:
    out_template = str(outdir / "video.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bv*+ba/b",
        "--merge-output-format", "mp4",
        "-o", out_template,
        url,
    ]
    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError("Erreur yt-dlp:\n" + proc.stdout)
    return outdir / "video.mp4"

# --- couper la vidéo ---
def make_clip(src: Path, start: float, end: float, dest: Path, reencode: bool):
    duration = max(0.01, end - start)
    if reencode:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(src),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(dest)
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(src),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            str(dest)
        ]
    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError("Erreur ffmpeg:\n" + proc.stdout)

# --- Interface utilisateur ---
with st.form("clip_form"):
    url = st.text_input("URL YouTube", placeholder="https://www.youtube.com/watch?v=...")
    col1, col2 = st.columns(2)
    with col1:
        start_tc = st.text_input("Début", placeholder="1:23 | 01:02:03 | 75 | 1m15s")
    with col2:
        end_tc = st.text_input("Fin", placeholder="2:10 | 01:03:05 | 130 | 2m10s")
    reencode = st.toggle("Coupe précise (réencodage, plus lent)", value=False)
    submit = st.form_submit_button("Extraire l’extrait")

if submit:
    if not url or not start_tc or not end_tc:
        st.warning("Merci de remplir l’URL, le début et la fin.")
    else:
        try:
            start_s = parse_timecode(start_tc)
            end_s = parse_timecode(end_tc)
            if end_s <= start_s:
                st.error("Le timecode de fin doit être supérieur au début.")
            else:
                with st.spinner("Traitement en cours…"):
                    with tempfile.TemporaryDirectory() as td:
                        td_path = Path(td)
                        src = download_best(url, td_path)
                        out_file = td_path / "clip.mp4"
                        make_clip(src, start_s, end_s, out_file, reencode)

                        st.success("✅ Extrait prêt !")
                        st.video(str(out_file))
                        with open(out_file, "rb") as f:
                            st.download_button("⬇️ Télécharger le clip", f, file_name="clip.mp4", mime="video/mp4")
        except Exception as e:
            st.error(f"Erreur: {e}")
