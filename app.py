import streamlit as st
import tempfile
import subprocess
from pathlib import Path
import re
import imageio_ffmpeg

st.set_page_config(page_title="YouTube Clip Extractor", page_icon="✂️", layout="centered")
st.title("✂️ YouTube Clip Extractor")
st.caption("Colle une URL YouTube + timecodes de début/fin → télécharge ton extrait (MP4).")

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

def parse_timecode(tc: str) -> float:
    tc = tc.strip().lower()
    if any(x in tc for x in ("h", "m", "s")):
        m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?", tc)
        if not m: raise ValueError(f"Timecode invalide: {tc}")
        h = float(m.group(1) or 0); mi = float(m.group(2) or 0); s = float(m.group(3) or 0)
        return h*3600 + mi*60 + s
    if ":" in tc:
        parts = [float(p) for p in tc.split(":")]
        if len(parts) == 2: return parts[0]*60 + parts[1]
        if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
        raise ValueError(f"Timecode invalide: {tc}")
    return float(tc)

def run(cmd: list):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def download_best(url: str, outdir: Path, prefer_format: str, force_ipv4: bool, cookies_file: Path|None) -> Path:
    # Formats stables côté YouTube : on privilégie MP4/H.264 + AAC (évite le merging exotique)
    format_map = {
        "auto-mp4": "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "format-18": "18",  # 360p mp4 (ultra compatible)
    }
    chosen_format = format_map[prefer_format]

    cmd = [
        "yt-dlp",
        "--ffmpeg-location", FFMPEG_BIN,
        "-f", chosen_format,
        "--merge-output-format", "mp4",
        "-o", str(outdir / "video.%(ext)s"),
        "--no-mtime",
        "--geo-bypass",
        "-N", "1",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "-R", "5", "--retry-sleep", "2",
    ]
    if force_ipv4:
        cmd += ["--force-ipv4"]
    if cookies_file is not None:
        cmd += ["--cookies", str(cookies_file)]

    cmd += [url]
    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError("Erreur yt-dlp:\n" + proc.stdout)

    # récupérer le mp4 le plus récent
    mp4s = sorted(outdir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        raise RuntimeError("Téléchargement terminé mais aucun MP4 détecté. Logs yt-dlp :\n" + proc.stdout)
    return mp4s[0]

def make_clip(src: Path, start: float, end: float, dest: Path, reencode: bool):
    dur = max(0.01, end - start)
    if reencode:
        cmd = [FFMPEG_BIN, "-y", "-ss", f"{start:.3f}", "-i", str(src),
               "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-c:a", "aac", "-b:a", "192k", str(dest)]
    else:
        cmd = [FFMPEG_BIN, "-y", "-ss", f"{start:.3f}", "-i", str(src),
               "-t", f"{dur:.3f}", "-c", "copy", str(dest)]
    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError("Erreur ffmpeg:\n" + proc.stdout)

with st.form("clip_form"):
    url = st.text_input("URL YouTube", placeholder="https://www.youtube.com/watch?v=...")
    c1, c2 = st.columns(2)
    with c1:
        start_tc = st.text_input("Début", placeholder="1:23 | 01:02:03 | 75 | 1m15s")
    with c2:
        end_tc   = st.text_input("Fin",   placeholder="2:10 | 01:03:05 | 130 | 2m10s")
    reencode   = st.toggle("Coupe précise (réencodage, plus lent)", value=False)
    force_ipv4 = st.toggle("Forcer IPv4 (utile si 403)", value=True)
    prefer_format = st.selectbox("Profil de téléchargement",
                                 ["auto-mp4", "format-18"],
                                 help="auto-mp4: meilleure qualité MP4/H.264 possible. format-18: 360p très compatible.")
    cookies_upload = st.file_uploader("Cookies (facultatif) – fichier Netscape cookies.txt", type=["txt"])
    submit = st.form_submit_button("Extraire l’extrait")

if submit:
    if not url or not start_tc or not end_tc:
        st.warning("Merci de remplir l’URL, le début et la fin.")
    else:
        try:
            s = parse_timecode(start_tc); e = parse_timecode(end_tc)
            if e <= s:
                st.error("Le timecode de fin doit être supérieur au début.")
            else:
                with st.spinner("Traitement en cours…"):
                    with tempfile.TemporaryDirectory() as td:
                        tdp = Path(td)
                        cookies_path = None
                        if cookies_upload is not None:
                            cookies_path = tdp / "cookies.txt"
                            cookies_path.write_bytes(cookies_upload.read())

                        src = download_best(url, tdp, prefer_format, force_ipv4, cookies_path)
                        out_file = tdp / "clip.mp4"
                        make_clip(src, s, e, out_file, reencode)

                        st.success("✅ Extrait prêt !")
                        st.video(str(out_file))
                        with open(out_file, "rb") as f:
                            st.download_button("⬇️ Télécharger le clip", f, file_name="clip.mp4", mime="video/mp4")
        except Exception as e:
            st.error(f"Erreur: {e}")
            st.caption("Si 403 persiste : essaie ‘format-18’, force IPv4, ou fournis un cookies.txt (vidéos restreintes).")
