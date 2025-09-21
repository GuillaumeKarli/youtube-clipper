import streamlit as st
import tempfile
import subprocess
from pathlib import Path
import re
import traceback
import hashlib, time, os, shutil, zipfile

# ================= UI de base =================
st.set_page_config(page_title="YouTube Clip Extractor", page_icon="✂️", layout="centered")
st.title("✂️ YouTube Clip Extractor")
st.caption("Colle une URL YouTube + timecodes, génère un ou plusieurs extraits, qualité au choix, sans retélécharger inutilement.")

# ================= Cache persistant (durée de vie du conteneur) =================
# Sur Streamlit Cloud gratuit, /tmp est écrivable; il est vidé aux redéploiements/redémarrages.
CACHE_DIR = Path("/tmp/cache_sources")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ================= FFMPEG (robuste) =================
FFMPEG_BIN = None
def get_ffmpeg_path():
    """Préférence au binaire d'imageio-ffmpeg, sinon 'ffmpeg' du PATH."""
    global FFMPEG_BIN
    if FFMPEG_BIN:
        return FFMPEG_BIN
    try:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG_BIN = "ffmpeg"
    return FFMPEG_BIN

# ================= Diagnostics + purge cache =================
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
            for p in CACHE_DIR.glob("*.mkv"):
                p.unlink(missing_ok=True)
            if "dl_cache" in st.session_state:
                st.session_state.dl_cache.clear()
            st.success("Cache vidé.")
        except Exception as e:
            st.error(f"Impossible de vider le cache : {e}")

# ================= Mémo en session =================
if "dl_cache" not in st.session_state:
    # key -> {"url","quality","force_ipv4","cookies_sha1","path","size","ts"}
    st.session_state.dl_cache = {}
CACHE_MAX = 8

def sha1_file(path: Path|None) -> str:
    if not path or not path.exists():
        return ""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
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
    if not os.path.exists(e["path"]):
        st.session_state.dl_cache.pop(key, None)
        return None
    e["ts"] = time.time()
    return e

# Reconstitution minimale du mémo depuis le disque (après redémarrage)
for p in sorted(CACHE_DIR.glob("*.*"), key=lambda x: x.stat().st_mtime, reverse=True):
    key = p.stem
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

# ================= Utils =================
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

def parse_segments_lines(text: str):
    """
    Lignes 'de - à' (mêmes formats que parse_timecode). Ex:
      1:21:06 - 1:23:52
      75 - 130
      1m15s - 2m10s
    """
    segs = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " - " in line:
            a, b = line.split(" - ", 1)
        elif "-" in line:
            a, b = line.split("-", 1)
        elif "→" in line:
            a, b = line.split("→", 1)
        elif " to " in line:
            a, b = line.split(" to ", 1)
        else:
            raise ValueError(f"Ligne invalide (attendu 'de - à'): {line}")
        s = parse_timecode(a.strip())
        e = parse_timecode(b.strip())
        if e <= s:
            raise ValueError(f"Fin <= début pour la ligne: {line}")
        segs.append((s, e))
    if not segs:
        raise ValueError("Aucun extrait détecté.")
    return segs

# ================= Téléchargement (jauge + qualité) =================
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
    # Choix "Original" = meilleur flux (VP9/AV1 possible), fusion en MKV (aucun ré-encodage).
    quality_map = {
        "Original (max, MKV)": {
            "format": "bv*+ba/b",
            "merge":  "mkv",
            "exts":   [".mkv", ".webm", ".mp4"],
        },
        "Haute (1080p max)": {
            "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "merge":  "mp4",
            "exts":   [".mp4"],
        },
        "Moyenne (720p max)": {
            "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "merge":  "mp4",
            "exts":   [".mp4"],
        },
        "Basse (360p)": {
            "format": "bestvideo[height<=360]+bestaudio/best[height<=360]",
            "merge":  "mp4",
            "exts":   [".mp4"],
        },
    }
    chosen = quality_map[quality]
    chosen_format = chosen["format"]
    merge_format  = chosen["merge"]
    expected_exts = chosen["exts"]

    cmd = [
        "yt-dlp",
        "--newline",
        "-f", chosen_format,
        "--merge-output-format", merge_format,
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
        if len(last_lines) > 10: last_lines.pop(0)
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

    # Cherche le fichier fusionné avec l'extension attendue
    found = []
    for ext in expected_exts:
        found += list(outdir.glob(f"*{ext}"))
    if not found:
        # fallback: prends le plus récent quel que soit l'ext
        found = list(outdir.glob("*.*"))
    if not found:
        raise RuntimeError("Téléchargement terminé mais aucun fichier détecté.")
    out = sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    progress_bar.progress(100)
    status_text.write("Téléchargement terminé ✅")
    return out

# ================= Cache wrapper =================
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
    # extension dépend du profil qualité (mkv pour "Original")
    ext = ".mkv" if quality_choice.startswith("Original") else ".mp4"
    cache_path = CACHE_DIR / f"{key}{ext}"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        status_text.markdown("**Cache HIT** · 🗂️ Source trouvée — pas de nouveau téléchargement.")
        progress_bar.progress(100)
        entry = {
            "url": url, "quality": quality_choice, "force_ipv4": force_ipv4,
            "cookies_sha1": csha, "path": str(cache_path),
            "size": cache_path.stat().st_size, "ts": time.time()
        }
        add_to_cache(key, entry)
        log_area.code(f"[cache] {entry['url']} ({entry['quality']}, {entry['size']//1024//1024} MB)", language="text")
        return cache_path

    status_text.markdown("**Cache MISS** · Téléchargement de la source…")
    tmp_src = download_with_progress(
        url, outdir, quality_choice, force_ipv4, cookies_path,
        progress_bar=progress_bar, status_text=status_text, log_area=log_area
    )
    # normalise l'extension cache (mkv pour Original, mp4 sinon)
    try:
        shutil.move(str(tmp_src), str(cache_path))
    except Exception:
        shutil.copy2(str(tmp_src), str(cache_path))

    entry = {
        "url": url, "quality": quality_choice, "force_ipv4": force_ipv4,
        "cookies_sha1": csha, "path": str(cache_path),
        "size": cache_path.stat().st_size, "ts": time.time(),
    }
    add_to_cache(key, entry)
    return cache_path

# ================= Découpe (avec contrôle de poids) =================
def clip_with_progress(
    src: Path,
    start: float,
    end: float,
    dest: Path,
    reencode: bool,
    progress_bar,
    status_text,
    log_area,
    target_height: int | None = None,
    crf: int = 23
):
    duration = max(0.01, end - start)
    ff = get_ffmpeg_path()

    base = [ff, "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}"]
    if reencode:
        base += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-c:a", "aac", "-b:a", "160k"]
        if target_height:
            base += ["-vf", f"scale=-2:{target_height}"]
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
        if len(last_lines) > 10: last_lines.pop(0)
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

# ================= Concat (montage) =================
def concat_videos_ffmpeg(inputs: list[Path], dest: Path, target_height: int|None = None, crf: int = 23):
    ff = get_ffmpeg_path()
    cmd = [ff, "-y"]
    for p in inputs:
        cmd += ["-i", str(p)]
    n = len(inputs)
    scale = f",scale=-2:{target_height}" if target_height else ""
    filt = ""
    for i in range(n):
        filt += f"[{i}:v]fps=30{scale},format=yuv420p[v{i}];"
        filt += f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{i}];"
    v_in = "".join([f"[v{i}]" for i in range(n)])
    a_in = "".join([f"[a{i}]" for i in range(n)])
    filt += f"{v_in}{a_in}concat=n={n}:v=1:a=1[v][a]"
    cmd += [
        "-filter_complex", filt,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-c:a", "aac", "-b:a", "160k",
        str(dest)
    ]
    proc = popen_stream(cmd)
    for _ in proc.stdout:
        pass
    if proc.wait() != 0:
        raise RuntimeError("Echec concaténation ffmpeg.")

# ================= Mémo visuel =================
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

# ================= UI principale =================
try:
    with st.form("clip_form"):
        default_url = st.session_state.get("prefill_url", "")
        url = st.text_input("URL YouTube", value=default_url, placeholder="https://www.youtube.com/watch?v=...")

        batch_mode = st.toggle("Plusieurs extraits (batch)", value=False,
                               help="Colle plusieurs lignes 'début - fin' pour générer plusieurs clips.")
        if not batch_mode:
            c1, c2 = st.columns(2)
            with c1:
                start_tc = st.text_input("Début", placeholder="1:23 | 01:02:03 | 75 | 1m15s")
            with c2:
                end_tc   = st.text_input("Fin",   placeholder="2:10 | 01:03:05 | 130 | 2m10s")
        else:
            seg_text = st.text_area(
                "Extraits (un par ligne, format 'début - fin')",
                value="1:21:06 - 1:23:52\n1:55:00 - 1:56:20",
                height=120
            )

        force_ipv4 = st.toggle("Forcer IPv4 (utile si 403)", value=True)

        quality_choice = st.selectbox(
            "Qualité vidéo (téléchargement)",
            ["Original (max, MKV)", "Haute (1080p max)", "Moyenne (720p max)", "Basse (360p)"],
            index=0,
            help="“Original (max, MKV)” = meilleure qualité possible (peut être 4K/VP9/AV1)."
        )

        # Profils de sortie
        output_profile = st.selectbox(
            "Poids du clip (sortie)",
            [
                "Qualité source (pas de réencodage)",
                "Compressé 720p (recommandé)",
                "Compressé 480p (léger)",
                "Auto (copie rapide si possible)"
            ],
            index=0,
            help="‘Qualité source’ = zéro perte (copie), coupe aux keyframes; 'Compressé' = plus léger et stable."
        )

        reencode_precise = st.toggle("Coupe précise (réencodage millimétré)", value=False)

        cookies_upload = st.file_uploader("Cookies (facultatif) – fichier Netscape cookies.txt", type=["txt"])
        submit = st.form_submit_button("Générer")

    if submit:
        if not url or (not batch_mode and (not start_tc or not end_tc)) or (batch_mode and not seg_text.strip()):
            st.warning("Merci de remplir l’URL et les timecodes.")
        else:
            try:
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

                    # 1) Télécharger (avec cache)
                    src = get_or_download(
                        url, tdp, quality_choice, force_ipv4, cookies_path,
                        progress_bar=dl_bar, status_text=dl_status, log_area=log_area
                    )

                    # 2) Paramètres de sortie
                    if output_profile == "Qualité source (pas de réencodage)":
                        out_reencode = False
                        out_target_h = None
                        out_crf = 18
                    elif output_profile == "Compressé 720p (recommandé)":
                        out_reencode = True
                        out_target_h = 720
                        out_crf = 23
                    elif output_profile == "Compressé 480p (léger)":
                        out_reencode = True
                        out_target_h = 480
                        out_crf = 24
                    else:  # Auto (copie rapide si possible)
                        out_reencode = reencode_precise
                        out_target_h = None
                        out_crf = 18

                    # 3) Générer un ou plusieurs extraits
                    generated_files = []

                    if not batch_mode:
                        s = parse_timecode(start_tc); e = parse_timecode(end_tc)
                        if e <= s:
                            st.error("Le timecode de fin doit être supérieur au début.")
                            st.stop()
                        out_file = tdp / ("clip" + (".mkv" if src.suffix.lower()==".mkv" and not out_reencode else ".mp4"))
                        clip_with_progress(
                            src, s, e, out_file, out_reencode,
                            progress_bar=clip_bar, status_text=clip_status, log_area=log_area,
                            target_height=out_target_h, crf=out_crf
                        )
                        generated_files.append((out_file.name, out_file))
                    else:
                        try:
                            segments = parse_segments_lines(seg_text)
                        except Exception as err:
                            st.error(f"Erreur de parsing des extraits: {err}")
                            st.stop()

                        st.write(f"**{len(segments)} extraits** à générer…")
                        global_bar = st.progress(0)
                        for idx, (s_val, e_val) in enumerate(segments, start=1):
                            clip_status.write(f"Extrait {idx}/{len(segments)} — découpe en cours…")
                            out_file = tdp / (f"clip_{idx:02d}" + (".mkv" if src.suffix.lower()==".mkv" and not out_reencode else ".mp4"))
                            clip_with_progress(
                                src, s_val, e_val, out_file, out_reencode,
                                progress_bar=clip_bar, status_text=clip_status, log_area=log_area,
                                target_height=out_target_h, crf=out_crf
                            )
                            generated_files.append((out_file.name, out_file))
                            global_bar.progress(int(idx * 100 / len(segments)))

                    # 4) Sorties
                    st.success("✅ Extrait(s) prêt(s) !")

                    # Clips unitaires
                    for name, path in generated_files:
                        size_mb = os.path.getsize(path) // (1024 * 1024)
                        st.caption(f"{name} • ~{size_mb} MB")

                        # Aperçu: seulement si raisonnable et compatible navigateur
                        show_preview = (size_mb <= 150) and (path.suffix.lower() in [".mp4", ".m4v", ".webm"])
                        if show_preview:
                            st.video(str(path))
                        else:
                            st.info(f"Aperçu désactivé pour {name} (gros fichier ou format non lisible). Télécharge directement.")

                        with open(path, "rb") as f:
                            mime = "video/mp4" if path.suffix.lower() == ".mp4" else "video/x-matroska"
                            st.download_button(f"⬇️ Télécharger {name}", f, file_name=name, mime=mime)

                    # ZIP “tout télécharger”
                    if len(generated_files) > 1:
                        zip_path = tdp / "extraits.zip"
                        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                            for name, path in generated_files:
                                zf.write(path, arcname=name)
                        with open(zip_path, "rb") as f:
                            st.download_button("⬇️ Télécharger tous les extraits (.zip)", f, file_name="extraits.zip", mime="application/zip")

                    # Montage concat
                    if len(generated_files) > 1:
                        st.markdown("---")
                        st.subheader("Assembler en un seul fichier")
                        assemble = st.toggle("Créer un montage continu (concat)", value=False,
                                             help="Ré-encode pour assurer la compatibilité. Plus stable.")
                        if assemble:
                            target_h_for_concat = out_target_h or (720 if output_profile.startswith("Compressé") else 720)
                            crf_for_concat = max(22, (out_crf if out_reencode else 23))
                            concat_out = tdp / "montage.mp4"
                            with st.spinner("Assemblage des extraits…"):
                                concat_videos_ffmpeg([p for _, p in generated_files], concat_out,
                                                     target_height=target_h_for_concat, crf=crf_for_concat)
                            st.success("Montage prêt ✅")
                            size_mb = os.path.getsize(concat_out) // (1024 * 1024)
                            st.caption(f"montage.mp4 • ~{size_mb} MB")
                            if size_mb <= 150:
                                st.video(str(concat_out))
                            with open(concat_out, "rb") as f:
                                st.download_button("⬇️ Télécharger le montage", f, file_name="montage.mp4", mime="video/mp4")

            except Exception as e:
                st.error(f"Erreur: {e}")
                st.exception(e)

except Exception:
    st.error("Erreur au démarrage de l'app.")
    st.code("".join(traceback.format_exc()))
