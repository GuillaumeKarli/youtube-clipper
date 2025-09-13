import streamlit as st
import tempfile
import subprocess
from pathlib import Path
import re
import traceback

# --- ffmpeg: résolution tolérante ---
FFMPEG_BIN = None
def get_ffmpeg_path():
    global FFMPEG_BIN
    if FFMPEG_BIN:
        return FFMPEG_BIN
    try:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # fallback sur ffmpeg du PATH (Streamlit Cloud en a souvent un)
        FFMPEG_BIN = "ffmpeg"
    return FFMPEG_BIN

# --- petits utilitaires ---
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

def popen_stream(cmd: list):
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

# --- UI de base (on met vite un cadre, puis on affiche Diagnostics) ---
st.set_page_config(page_title="YouTube Clip Extractor", page_icon="✂️", layout="centered")
st.title("✂️ YouTube Clip Extractor")
st.caption("Colle une URL YouTube + timecodes de début/fin → télécharge ton extrait (MP4).")

with st.sidebar:
    st.markdown("### 🧪 Diagnostics")
    try:
        import sys, shutil
        # versions
        st.write("Python:", sys.version.split()[0])
        yv = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        st.write("yt-dlp:", (yv.stdout or yv.stderr).strip())
        # ffmpeg
        ff = get_ffmpeg_path()
        st.write("ffmpeg path:", ff)
        which_ff = shutil.which("ffmpeg")
        st.write("ffmpeg in PATH:", bool(which_ff))
    except Exception as e:
        st.error(f"Diag error: {e}")

# ========================
# Tout le reste dans un try pour afficher les traces dans l'UI
# ========================
try:
    # ----------------- Téléchargement (avec jauge + qualité) -----------------
    def download_with_progress(
        url: str,
        outdir: Path,
        quality: str,
        force_ipv4: bool,
        cookies_file: Path | None,
        progress_bar,
        status_text,
        log_area
    ) -> Path:
        quality_map = {
            "Haute (1080p max)":  "bestvideo[height<=1080][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "Moyenne (720p max)": "bestvideo[height<=720][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "Basse (360p)":       "bestvideo[height<=360][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/best",
        }
        chosen_format = quality_map[quality]

        cmd = [
            "yt-dlp",
            "--newline",
            "-f", chosen_format,
            "--merge-output-format", "mp4",
            "-o", str(outdir / "video.%(ext)s"),
            "--no-mtime",
            "--geo-bypass",
            "-N", "1",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "-R", "5", "--retry-sleep", "2",
            url
        ]

        # On passe ffmpeg-location seulement si on a un chemin explicite (évite des Plantages rares)
        ff = get_ffmpeg_path()
        if ff and ff != "ffmpeg":
            cmd[1:1] = ["--ffmpeg-location", ff]

        if force_ipv4:
            cmd[cmd.index(url):cmd.index(url)] = ["--force-ipv4"]
        if cookies_file is not None:
            cmd[cmd.index(url):cmd.index(url)] = ["--cookies", str(cookies_file)]

        proc = popen_stream(cmd)
        pct = 0
        progress_bar.progress(pct)
        last_lines = []
        pct_re = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")

        for line in proc.stdout:
            line = line.rstrip()
            if not line: continue
            last_lines.append(line)
            if len(last_lines) > 8: last_lines.pop(0)
            log_area.code("\n".join(last_lines), language="text")
            m = pct_re.search(line)
            if m:
                try:
                    pct = min(100, max(0, float(m.group(1))))
                    progress_bar.progress(int(pct))
                    status_text.write(f"Téléchargement… {pct:.1f}%")
                except Exception:
                    pass

        retcode = proc.wait()
        if retcode != 0:
            raise RuntimeError("Erreur yt-dlp:\n" + "\n".join(last_lines))

        mp4s = sorted(outdir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            raise RuntimeError("Téléchargement terminé mais aucun MP4 détecté.")
        progress_bar.progress(100)
        status_text.write("Téléchargement terminé ✅")
        return mp4s[0]

    # ----------------- Coupe (avec jauge) -----------------
    def clip_with_progress(
        src: Path,
        start: float,
        end: float,
        dest: Path,
        reencode: bool,
        progress_bar,
        status_text,
        log_area
    ):
        duration = max(0.01, end - start)
        ff = get_ffmpeg_path()
        base = [ff, "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}"]
        if reencode:
            base += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k"]
        else:
            base += ["-c", "copy"]

        cmd = base + ["-progress", "pipe:1", "-nostats", "-loglevel", "warning", str(dest)]
        proc = popen_stream(cmd)

        last_lines = []
        progress_bar.progress(0)
        status_text.write("Découpe en cours… 0%")

        out_time_re = re.compile(r"out_time_ms=(\d+)")
        pct = 0

        for line in proc.stdout:
            line = line.strip()
            if not line: continue
            last_lines.append(line)
            if len(last_lines) > 8: last_lines.pop(0)
            log_area.code("\n".join(last_lines), language="text")
            m = out_time_re.match(line)
            if m:
                try:
                    out_ms = int(m.group(1))
                    cur = out_ms / 1_000_000.0
                    pct = min(100, max(0, (cur / duration) * 100))
                    progress_bar.progress(int(pct))
                    status_text.write(f"Découpe en cours… {pct:.0f}%")
                except Exception:
                    pass

        retcode = proc.wait()
        if retcode != 0:
            raise RuntimeError("Erreur ffmpeg:\n" + "\n".join(last_lines))

        progress_bar.progress(100)
        status_text.write("Découpe terminée ✅")

    # ----------------- UI principale -----------------
    with st.form("clip_form"):
        url = st.text_input("URL YouTube", placeholder="https://www.youtube.com/watch?v=...")
        c1, c2 = st.columns(2)
        with c1:
            start_tc = st.text_input("Début", placeholder="1:23 | 01:02:03 | 75 | 1m15s")
        with c2:
            end_tc   = st.text_input("Fin",   placeholder="2:10 | 01:03:05 | 130 | 2m10s")
        reencode   = st.toggle("Coupe précise (réencodage, plus lent)", value=False)
        force_ipv4 = st.toggle("Forcer IPv4 (utile si 403)", value=True)
        quality_choice = st.selectbox(
            "Qualité vidéo",
            ["Haute (1080p max)", "Moyenne (720p max)", "Basse (360p)"],
            help="Choisis la résolution maximale souhaitée."
        )
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
                    with tempfile.TemporaryDirectory() as td:
                        tdp = Path(td)
                        cookies_path = None
                        if cookies_upload is not None:
                            cookies_path = tdp / "cookies.txt"
                            cookies_path.write_bytes(cookies_upload.read())

                        st.subheader("Progression")
                        dl_status = st.empty()
                        dl_bar = st.progress(0)
                        clip_status = st.empty()
                        clip_bar = st.progress(0)
                        st.subheader("Logs")
                        log_area = st.empty()

                        src = download_with_progress(
                            url, tdp, quality_choice, force_ipv4, cookies_path,
                            progress_bar=dl_bar, status_text=dl_status, log_area=log_area
                        )

                        st.download_button(
                            "⬇️ Télécharger la vidéo complète (source)",
                            data=open(src, "rb"),
                            file_name="video_complete.mp4",
                            mime="video/mp4",
                        )

                        out_file = tdp / "clip.mp4"
                        clip_with_progress(
                            src, s, e, out_file, reencode,
                            progress_bar=clip_bar, status_text=clip_status, log_area=log_area
                        )

                        st.success("✅ Extrait prêt !")
                        st.video(str(out_file))
                        with open(out_file, "rb") as f:
                            st.download_button("⬇️ Télécharger le clip", f, file_name="clip.mp4", mime="video/mp4")

            except Exception as e:
                st.error(f"Erreur: {e}")
                st.exception(e)  # affiche la stacktrace utile

except Exception as boot_e:
    st.error("Erreur au démarrage de l'app.")
    st.code("".join(traceback.format_exc()))
