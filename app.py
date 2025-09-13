import streamlit as st
import tempfile
import subprocess
from pathlib import Path
import re
import os
import imageio_ffmpeg

st.set_page_config(page_title="YouTube Clip Extractor", page_icon="✂️", layout="centered")
st.title("✂️ YouTube Clip Extractor")
st.caption("Colle une URL YouTube + timecodes de début/fin → télécharge ton extrait (MP4).")

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()  # binaire ffmpeg pour le cloud

# ---------- utils ----------
def parse_timecode(tc: str) -> float:
    tc = tc.strip().lower()
    if any(x in tc for x in ("h", "m", "s")):
        m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?", tc)
        if not m:
            raise ValueError(f"Timecode invalide: {tc}")
        h = float(m.group(1) or 0); mi = float(m.group(2) or 0); s = float(m.group(3) or 0)
        return h*3600 + mi*60 + s
    if ":" in tc:
        parts = [float(p) for p in tc.split(":")]
        if len(parts) == 2:  # MM:SS
            return parts[0]*60 + parts[1]
        if len(parts) == 3:  # HH:MM:SS
            return parts[0]*3600 + parts[1]*60 + parts[2]
        raise ValueError(f"Timecode invalide: {tc}")
    return float(tc)

def run(cmd: list):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# ---------- yt-dlp ----------
def download_best(url: str, outdir: Path, compat: bool, cookies_file: Path | None) -> Path:
    out_template = str(outdir / "video.%(ext)s")
    base_cmd = [
        "yt-dlp",
        "--ffmpeg-location", FFMPEG_BIN,          # <-- important pour la fusion
        "-f", "bv*+ba/b",
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--no-mtime",
    ]
    # Mode compatibilité pour contourner certains 403/throttling
    if compat:
        base_cmd += [
            "--force-ipv4",
            "-N", "1",
            "--extractor-args", "youtube:player_client=android",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        ]
    # Cookies (facultatif) si vidéo restreinte (âge/consentement)
    if cookies_file is not None:
        base_cmd += ["--cookies", str(cookies_file)]

    cmd = base_cmd + [url]
    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError("Erreur yt-dlp:\n" + proc.stdout)
    mp4 = outdir / "video.mp4"
    if not mp4.exists():
        # yt-dlp peut sortir .m4a/.webm dans certains cas, sécurisons
        # on cherche le dernier .mp4 créé
        mp4s = sorted(outdir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            raise RuntimeError("Téléchargement terminé mais aucun MP4 fusionné. Logs yt-dlp :\n" + proc.stdout)
        mp4 = mp4s[0]
    return mp4

# ---------- ffmpeg ----------
def make_clip(src: Path, start: float, end: float, dest: Path, reencode: bool):
    duration = max(0.01, end - start)
    if reencode:
        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{start:.3f}",
            "-i", str(src),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(dest)
        ]
    else:
        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{start:.3f}",
            "-i", str(src),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            str(dest)
        ]
    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError("Erreur ffmpeg:\n" + proc.stdout)

# ---------- UI ----------
with st.form("clip_form"):
    url = st.text_input("URL YouTube", placeholder="https://www.youtube.com/watch?v=...")
    c1, c2 = st.columns(2)
    with c1:
        start_tc = st.text_input("Début", placeholder="1:23 | 01:02:03 | 75 | 1m15s")
    with c2:
        end_tc = st.text_input("Fin", placeholder="2:10 | 01:03:05 | 130 | 2m10s")
    reencode = st.toggle("Coupe précise (réencodage, plus lent)", value=False)
    compat = st.toggle("Mode compatibilité (si erreur 403)", value=True)
    cookies_upload = st.file_uploader("Cookies (facultatif) – fichier Netscape cookies.txt", type=["txt"])
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

                        # Enregistre les cookies si fournis
                        cookies_path = None
                        if cookies_upload is not None:
                            cookies_path = td_path / "cookies.txt"
                            cookies_path.write_bytes(cookies_upload.read())

                        src = download_best(url, td_path, compat=compat, cookies_file=cookies_path)
                        out_file = td_path / "clip.mp4"
                        make_clip(src, start_s, end_s, out_file, reencode)

                        st.success("✅ Extrait prêt !")
                        st.video(str(out_file))
                        with open(out_file, "rb") as f:
                            st.download_button("⬇️ Télécharger le clip", f, file_name="clip.mp4", mime="video/mp4")
        except Exception as e:
            st.error(f"Erreur: {e}")
            st.caption("Astuce: active le ‘Mode compatibilité’, ou fournis un cookies.txt si la vidéo est restreinte.")
