import streamlit as st, subprocess, shutil, sys

st.set_page_config(page_title="Diag", layout="centered")
st.title("✅ Smoke test")

st.write("Python:", sys.version.split()[0])
st.write("ffmpeg in PATH:", shutil.which("ffmpeg"))
yt = subprocess.run(["yt-dlp","--version"], capture_output=True, text=True)
st.write("yt-dlp:", (yt.stdout or yt.stderr).strip())

st.success("Si tu vois cette page, l'environnement est OK.")
