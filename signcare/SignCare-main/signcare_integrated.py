import streamlit as st
import speech_recognition as sr
import moviepy.editor as mpy
import numpy as np
import cv2
import mediapipe as mp
import threading
import time
import os
import glob
import tensorflow as tf
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# ------------------- NLTK (offline — no downloads attempted) -------------------
import nltk
NLTK_DIR = r"C:\Users\K.SRI KRISHNA REDDY\nltk_data"
nltk.data.path.append(NLTK_DIR)
# Only download if the data is genuinely missing — works offline if already present
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, download_dir=NLTK_DIR, quiet=True)
        except Exception:
            pass  # Offline and missing — will fall back to simple split below

try:
    from nltk.corpus import stopwords as nltk_sw
    from nltk.tokenize import word_tokenize as nltk_word_tokenize
    STOP_WORDS = set(nltk_sw.words("english"))
    def word_tokenize(text):
        return nltk_word_tokenize(text, preserve_line=True)
except Exception:
    # Graceful offline fallback — simple whitespace tokenizer
    STOP_WORDS = {
        "i","me","my","we","our","you","your","he","she","it","they","them",
        "is","are","was","were","be","been","being","have","has","had","do",
        "does","did","will","would","could","should","may","might","shall",
        "a","an","the","and","but","or","so","if","in","on","at","to","for",
        "of","with","by","from","as","this","that","these","those","not","no",
    }
    def word_tokenize(text):
        return text.lower().split()
# ------------------- END NLTK -------------------

from deep_translator import GoogleTranslator

VideoFileClip          = mpy.VideoFileClip
concatenate_videoclips = mpy.concatenate_videoclips

# Dynamically built from whatever .mp4 files exist in dataset/
# No need to ever update this manually when you add new videos.
def get_available_words() -> set:
    return {
        os.path.splitext(os.path.basename(f))[0].lower()
        for f in glob.glob("dataset/*.mp4")
    }

list_of_words = get_available_words()  # loaded once at startup


# ==============================================================
#  Model loader — cached, runs on GPU if available
# ==============================================================
@st.cache_resource
def load_gesture_model():
    """Load model and labels. Never call st.* here — cached functions
    cannot replay st calls made against layout blocks created outside them."""
    try:
        model  = load_model("model/gesture_model.h5")
        labels = np.load("model/labels.npy")

        # Warm-up call so first real inference isn't slow
        dummy = np.zeros((1, 126), dtype=np.float32)
        model(dummy, training=False)

        gpu_list = tf.config.list_physical_devices("GPU")
        device   = "GPU" if gpu_list else "CPU"
        # Return status string instead of calling st.* directly
        return model, labels, f"Gesture model loaded on {device} ✔", None
    except Exception as e:
        return None, None, None, str(e)


def get_gesture_model():
    """Thin wrapper: loads model and surfaces toasts/errors via st.*"""
    model, labels, ok_msg, err_msg = load_gesture_model()
    if ok_msg:
        st.toast(ok_msg, icon="🤟")
    if err_msg:
        st.error(f"Could not load gesture model: {err_msg}")
    return model, labels


# ==============================================================
#  WebRTC Video Processor
# ==============================================================
class GestureProcessor(VideoProcessorBase):

    # Process every Nth frame — 2 = skip every other frame (plenty for 30fps)
    PROCESS_EVERY_N = 2

    def __init__(self):
        self.model, self.labels = get_gesture_model()
        self._mp_hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,   # slightly lower = faster detection
            min_tracking_confidence=0.5,
        )
        self._mp_draw  = mp.solutions.drawing_utils

        self._lock               = threading.Lock()
        self._prediction_history: list = []
        self._sentence:           list = []
        self._last_added          = ""
        self.current_text         = ""
        self._frame_count         = 0       # for frame skipping
        self._last_landmarks      = None    # reuse landmarks on skipped frames

    @property
    def sentence(self) -> list:
        with self._lock:
            return list(self._sentence)

    def clear_sentence(self):
        with self._lock:
            self._sentence.clear()
            self._last_added  = ""
            self.current_text = ""
            self._prediction_history.clear()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self._frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        # ── Frame skipping: run MediaPipe + model only every Nth frame ──────
        should_process = (self._frame_count % self.PROCESS_EVERY_N == 0)

        landmarks = []
        text      = "No Hand"

        if should_process:
            rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self._mp_hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    self._mp_draw.draw_landmarks(
                        img, hand_lms,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )
                    for lm in hand_lms.landmark:
                        landmarks += [lm.x, lm.y, lm.z]

                landmarks = (landmarks + [0] * 126)[:126]
                self._last_landmarks = landmarks   # cache for skipped frames
            else:
                self._last_landmarks = None
        else:
            # Skipped frame — reuse last landmarks (no MediaPipe call)
            landmarks = self._last_landmarks or []
            # Still draw skeleton if we have hand results cached
            # (we skip re-drawing on skipped frames for speed)

        # ── Model inference (only when we have landmarks) ───────────────────
        if landmarks and self.model is not None:
            arr = np.array(landmarks, dtype=np.float32).reshape(1, 126)

            # Use model() directly — ~3–5x faster than model.predict() on GPU
            # because it skips Keras overhead (data conversion, progress bar, etc.)
            pred       = self.model(arr, training=False).numpy()
            classID    = int(np.argmax(pred))
            confidence = float(pred[0][classID])

            with self._lock:
                self._prediction_history.append((classID, confidence))
                if len(self._prediction_history) > 20:
                    self._prediction_history.pop(0)

                same_class = [p for p in self._prediction_history if p[0] == classID]
                high_conf  = [p for p in same_class if p[1] > 0.85]
                stable = len(high_conf) > 6

                if stable:
                    text = self.labels[classID]
                    if text != self._last_added:
                        self._sentence.append(text)
                        self._last_added = text
                else:
                    text = ""

                sentence_snap = list(self._sentence)

            self.current_text = text

        else:
            with self._lock:
                sentence_snap = list(self._sentence)

        # ── Annotate frame ───────────────────────────────────────────────────
        cv2.putText(img, f"Gesture : {text}",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, "Sentence: " + " ".join(sentence_snap),
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==============================================================
#  Speech-to-Text helpers
# ==============================================================
def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="en").translate(text).lower()
    except Exception:
        # Offline fallback — return as-is (already English or untranslated Telugu)
        st.warning("Translation unavailable offline — using original text.")
        return text.lower()


def translate_to_telugu(text: str) -> str:
    """Translate any text to Telugu. Returns original text on failure."""
    try:
        return GoogleTranslator(source="auto", target="te").translate(text)
    except Exception:
        return text


def preprocess_text(text: str) -> str:
    tokens = word_tokenize(text)
    return " ".join(w for w in tokens if w.isalnum() and w not in STOP_WORDS)


def play_video_for_words(play_seq: list):
    # Refresh from disk each time — picks up any newly added videos
    available = get_available_words()
    # Exclude combined_ files that were generated by this app
    available = {w for w in available if not w.startswith("combined_")}

    if not play_seq:
        st.warning("Nothing to play.")
        return

    if len(play_seq) == 1:
        word = play_seq[0]
        if word in available:
            try:
                st.video(open(f"dataset/{word}.mp4", "rb").read())
            except FileNotFoundError:
                st.error(f"Video not found: dataset/{word}.mp4")
        else:
            st.error(f"'{word}' is not in the ISL dataset.")
        return

    final_path = f"dataset/combined_{'_'.join(play_seq)}.mp4"

    # ── Cache hit: combined file already exists — skip re-rendering ──────────
    if os.path.exists(final_path):
        st.info("▶ Loaded from cache.")
        try:
            st.video(open(final_path, "rb").read())
        except Exception as e:
            st.error(f"Failed to load cached video: {e}")
        return

    # ── Cache miss: build the combined clip and save it ───────────────────────
    clips, missing = [], []
    for word in play_seq:
        if word in available:
            try:
                clips.append(VideoFileClip(f"dataset/{word}.mp4"))
            except Exception:
                missing.append(word)
        else:
            missing.append(word)

    if missing:
        st.warning(f"Words skipped (not in dataset): {', '.join(missing)}")
    if not clips:
        st.error("No valid clips to combine.")
        return

    concatenate_videoclips(clips, method="compose").write_videofile(
        final_path, codec="libx264", logger=None
    )
    try:
        st.video(open(final_path, "rb").read())
    except Exception as e:
        st.error(f"Failed to load combined video: {e}")


def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        st.info("🎙️ Listening… speak now.")
        audio = r.listen(source, timeout=10)

    try:
        # recognize_google requires internet — show clear error if offline
        raw = r.recognize_google(audio, language="te-IN")
        st.success(f"**Recognised:** {raw}")

        translated = translate_to_english(raw)
        st.info(f"**Translated:** {translated}")

        clean = preprocess_text(translated)
        st.write(f"**Preprocessed:** {clean}")

        seq = clean.split()
        if seq:
            play_video_for_words(seq)
        else:
            st.warning("No usable words after preprocessing.")

    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
    except sr.RequestError:
        st.error(
            "⚠️ Speech Recognition requires an internet connection. "
            "Please connect and try again."
        )


# ==============================================================
#  Streamlit UI
# ==============================================================
def main():
    st.set_page_config(page_title="SignCare", page_icon="🤟", layout="wide")
    st.title("🤟 SignCare")
    st.caption(
        "Virtual Gesture Communication for Hearing-Impaired — Telugu States Edition"
    )

    if "gesture_sentence" not in st.session_state:
        st.session_state.gesture_sentence = []

    mode = st.radio(
        "Select Mode",
        ["🎙️ Speech → ISL Video", "🤟 Gesture → Text (Live Webcam)"],
        horizontal=True,
    )
    st.divider()

    # ----------------------------------------------------------
    # MODE 1 — Speech → ISL Video
    # ----------------------------------------------------------
    if mode == "🎙️ Speech → ISL Video":
        st.subheader("Speech → ISL Video")
        st.write(
            "Speak in **Telugu or English**. "
            "The app translates, preprocesses, and plays the matching ISL video."
        )
        st.caption("⚠️ This mode requires an internet connection (Google Speech API + Translator).")
        if st.button("▶ Start Speech Recognition", use_container_width=True):
            speech_to_text()

    # ----------------------------------------------------------
    # MODE 2 — Gesture (fully offline after first load)
    # ----------------------------------------------------------
    elif mode == "🤟 Gesture → Text (Live Webcam)":
        st.subheader("Live Gesture Recognition")
        st.caption("✅ This mode works fully offline once the app has loaded.")

        col_cam, col_info = st.columns([3, 2])

        with col_cam:
            # ── Offline-safe ICE config ──────────────────────────────────────
            # No external STUN server needed when browser and server are on the
            # same machine (localhost) or same LAN. An empty iceServers list
            # tells WebRTC to use host candidates only — works fully offline.
            webrtc_ctx = webrtc_streamer(
                key="isl-gesture",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=GestureProcessor,
                rtc_configuration={
                    "iceServers": []          # ← empty = offline/LAN mode
                },
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        with col_info:
            st.markdown("### Live Output")
            gesture_box   = st.empty()
            sentence_box  = st.empty()
            translation_box = st.empty()
            clear_btn     = st.button("🗑 Clear Sentence")
            play_btn_slot = st.empty()
            status_box    = st.empty()

            if webrtc_ctx.video_processor:
                vp: GestureProcessor = webrtc_ctx.video_processor

                if clear_btn:
                    vp.clear_sentence()
                    st.session_state.gesture_sentence = []

                # ── Button created ONCE per script run — never inside the loop ──
                # Streamlit requires every widget key to appear exactly once.
                # The loop only reads vp.sentence; it never re-creates the button.
                play_pressed = play_btn_slot.button("▶ Play ISL Video", key="play_live")

                # ── Stable polling loop — updates placeholders in-place ──────
                prev_sentence: list = []
                prev_telugu:   str  = ""

                while webrtc_ctx.state.playing:
                    sentence = vp.sentence
                    current  = vp.current_text
                    st.session_state.gesture_sentence = sentence

                    gesture_box.markdown(f"**Current gesture:** `{current or '—'}`")
                    sentence_box.markdown(
                        "**Sentence (English):** "
                        + (" ".join(sentence) if sentence else "_none yet_")
                    )

                    # Only re-translate when sentence actually changes
                    if sentence != prev_sentence:
                        prev_telugu = translate_to_telugu(" ".join(sentence)) if sentence else ""
                        prev_sentence = list(sentence)

                    translation_box.markdown(
                        "**వాక్యం (Telugu):** "
                        + (prev_telugu if prev_telugu else "_none yet_")
                    )

                    # Handle play button press (captured before loop started)
                    if play_pressed and sentence:
                        play_video_for_words(
                            [w for w in sentence if w in list_of_words]
                        )
                        play_pressed = False  # consume — don't replay on next tick

                    time.sleep(0.4)

            else:
                status_box.info("Click **START** on the camera widget to begin.")

        if (not webrtc_ctx.state.playing
                and st.session_state.gesture_sentence):
            st.divider()
            final_english = " ".join(st.session_state.gesture_sentence)
            final_telugu  = translate_to_telugu(final_english)
            st.success(
                f"**Final Sentence (English):** {final_english}  \n"
                f"**చివరి వాక్యం (Telugu):** {final_telugu}"
            )
            if st.button("▶ Play ISL Video for Final Sentence"):
                play_video_for_words(
                    [w for w in st.session_state.gesture_sentence
                     if w in list_of_words]
                )


if __name__ == "__main__":
    main()