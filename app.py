import streamlit as st
import tempfile
import subprocess
from pathlib import Path
import re
import traceback
import hashlib, time, os, shutil

# ========= Config UI =========
st.set_page_config(page_title="YouTube Clip Extractor", page_icon="✂️", layout="centered")
st.title("✂️ YouTube Clip Extractor")
st.caption("Colle une URL YouTube + timecodes de début/fin → télécharge un extrait (MP4).")

# ========= Dossier de cache persistant (Cloud) =========
# IMPORTANT: /mount/data est l'espace persistant sur Streamlit Cloud
CACHE_DIR = Path("/tmp/cache_sources")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ========= FFMPEG (robuste) =========
FFMPEG_BIN = None
def get_ffmpeg_path():
    """Tente d'utiliser le binaire d'imageio-ffmpeg, sinon 'ffmpeg' du PATH."""
    global FFMPEG_BIN
    if FFMPEG_BIN:
        return FFMPEG_BIN
    try:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG_BIN = "ffmpeg"
    return FFMPEG_BIN

# ========= Diagnostics (sidebar) + purge cache =========
with st.sidebar:
    st.markdown("### 🧪 Diagnostics")
    try:
        import sys
        st.write("Python:", sys.version.split()[0])
        yv = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        st.write("yt-dlp:", (yv.stdout or yv.stderr).strip())
        ff = get_ffmpeg_path()
        st.write("ffmpeg path:", ff)
        st.write("ffmpeg in PATH:", bool(shutil.which("ffmpeg")))
    except Exception as e:
        st.error(f"Diag error: {e}")

    st.markdown("---")
    if st.button("🧹 Vider le cache des sources"):
        try:
            for p in CACHE_DIR.glob("*.mp4"):
                p.unlink(missing_ok=True)
            if "dl_cache" in st.session_state:
                st.session_state.dl_cache.clear()
            st.success("Cache vidé.")
        except Exception as e:
            st.error(f"Impossible de vider le cache : {e}")

# ========= Cache (session) pour mémo/LRU =========
if "dl_cache" not in st.session_state:
    # key -> {"url","quality","force_ipv4","cookies_sha1","path","size","ts"}
    st.session_state.dl_cache = {}
CACHE_MAX = 6

def sha1_file(path: Path|None) -> str:
    if not path or not path.exists():
        return ""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):  # 1 MB
            h.update(chunk)
    return h.hexdigest()

def make_cache_key(url: str, quality: str, force_ipv4: bool, cookies_sha1: str) -> str:
    raw = f"{url}|{quality}|{int(force_ipv4)}|{cookies_sha1}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def add_to_cache(key: str, entry: dict):
    cache = st.session_state.dl_cache
    cache[key] = entry
    if len(cache) > CACHE_MAX:
        oldest = min(cache.keys(), key=lambda k: cache[k]["ts"])
        cache.pop(oldest, None)

def get_from_cache(key: str) -> dict|None:
    e = st.session_state.dl_cache.get(key)
    if not e:
        return None
    if not os.path.exists(e["path"]):  # fichier égaré → invalider
        st.session_state.dl_cache.pop(key, None)
        return None
    e["ts"] = time.time()  # LRU touch
    return e

# ========= Recharger le mémo depuis le disque (patch n°3) =========
# Permet de revoir les sources même après restart/redeploy
for p in sorted(CACHE_DIR.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
    key = p.stem  # hash généré par make_cache_key
    if key not in st.session_state.dl_cache:
        st.session_state.dl_cache[key] = {
            "url": "(inconnue – clé="+key[:7]+")",
            "quality": "(?)",
            "force_ipv4": True,
            "cookies_sha1": "",
            "path": str(p),
            "size": p.stat().st_size,
            "ts": p.stat().st_mtime,
        }

# ========= Utilitaires =========
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

# ========= Téléchargement (jauge + qualité) =========
def download_with_progress(
    url: str,
    outdir: Path,
    quality: str,
    force_ipv4: bool,
    cookies_file: Path|None,
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

# ========= Wrapper cache (patch n°1 & n°2) =========
def get_or_download(
    url: str,
    outdir: Path,
    quality_choice: str,
    force_ipv4: bool,
    cookies_path: Path|None,
    progress_bar, status_text, log_area
) -> Path:
    csha = sha1_file(cookies_path)
    key  = make_cache_key(url, quality_choice, force_ipv4, csha)
    cache_mp4 = CACHE_DIR / f"{key}.mp4"

    # 1) Si le fichier existe déjà en persistant, on l'utilise
    if cache_mp4.exists() and cache_mp4.stat().st_size > 0:
        status_text.markdown("**Cache HIT** · 🗂️ Source trouvée — pas de nouveau téléchargement.")
        progress_bar.progress(100)
        entry = {
            "url": url, "quality": quality_choice, "force_ipv4": force_ipv4,
            "cookies_sha1": csha, "path": str(cache_mp4),
            "size": cache_mp4.stat().st_size, "ts": time.time()
        }
        add_to_cache(key, entry)
        log_area.code(f"[cache] {entry['url']} ({entry['quality']}, {entry['size']//1024//1024} MB)", language="text")
        return cache_mp4

    # 2) Sinon, on télécharge puis on copie vers le cache persistant
    status_text.markdown("**Cache MISS** · Téléchargement de la source…")
    tmp_src = download_with_progress(
        url, outdir, quality_choice, force_ipv4, cookies_path,
        progress_bar=progress_bar, status_text=status_text, log_area=log_area
    )
    try:
        shutil.move(str(tmp_src), str(cache_mp4))
    except Exception:
        shutil.copy2(str(tmp_src), str(cache_mp4))

    entry = {
        "url": url, "quality": quality_choice, "force_ipv4": force_ipv4,
        "cookies_sha1": csha, "path": str(cache_mp4),
        "size": cache_mp4.stat().st_size, "ts": time.time(),
    }
    add_to_cache(key, entry)
    return cache_mp4

# ========= Découpe (jauge) =========
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

# ========= Mémo visuel des sources =========
with st.expander("🗂️ Mémo des sources déjà téléchargées"):
    items = sorted(st.session_state.dl_cache.values(), key=lambda e: e["ts"], reverse=True)
    if not items:
        st.caption("Aucune source en cache pour l’instant.")
    else:
        for i, e in enumerate(items, 1):
            cols = st.columns([6,2,2,2])
            with cols[0]:
                st.write(f"**{i}.** {e['url']}")
                st.caption(f"{e['quality']} • {e['size']//1024//1024} MB • {time.strftime('%H:%M:%S', time.localtime(e['ts']))}")
            with cols[1]:
                if st.button("Recharger URL", key=f"reuse_{i}"):
                    st.session_state["prefill_url"] = e["url"]
            with cols[2]:
                st.write("IPv4:", "Oui" if e["force_ipv4"] else "Non")
            with cols[3]:
                st.write("Cookies:", "Oui" if e["cookies_sha1"] else "Non")

# ========= UI principale =========
try:
    with st.form("clip_form"):
        default_url = st.session_state.get("prefill_url", "")
        url = st.text_input("URL YouTube", value=default_url, placeholder="https://www.youtube.com/watch?v=...")
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

                        # Téléchargement (avec cache persistant /mount/data)
                        src = get_or_download(
                            url, tdp, quality_choice, force_ipv4, cookies_path,
                            progress_bar=dl_bar, status_text=dl_status, log_area=log_area
                        )

                        # Télécharger la vidéo source complète (optionnel)
                        st.download_button(
                            "⬇️ Télécharger la vidéo complète (source)",
                            data=open(src, "rb"),
                            file_name="video_complete.mp4",
                            mime="video/mp4",
                        )

                        # Découpe
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
                st.exception(e)

except Exception:
    st.error("Erreur au démarrage de l'app.")
    st.code("".join(traceback.format_exc()))
