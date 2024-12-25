import sys
import os
from axengine import InferenceSession
import numpy as np
import librosa
from typing import Tuple
import soundfile as sf
import base64
import zhconv
from languages import *


WHISPER_N_MELS      = 80
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_FFT       = 480
WHISPER_HOP_LENGTH  = 160

WHISPER_SOT           = 50258
WHISPER_EOT           = 50257
WHISPER_BLANK         = 220
WHISPER_NO_TIMESTAMPS = 50363
WHISPER_NO_SPEECH     = 50362
WHISPER_TRANSLATE     = 50358
WHISPER_TRANSCRIBE    = 50359
WHISPER_VOCAB_SIZE    = 51865
WHISPER_N_TEXT_CTX    = 448

NEG_INF = float("-inf")
SOT_SEQUENCE = np.array([WHISPER_SOT,WHISPER_SOT + 1 + tuple(WHISPER_LANGUAGES).index("zh"),WHISPER_TRANSCRIBE,WHISPER_NO_TIMESTAMPS], dtype=np.int32)
WHISPER_N_TEXT_STATE_MAP = {
    "tiny": 384,
    "small": 768
}


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=WHISPER_SAMPLE_RATE)
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def load_models(model_path, model_type):
    encoder_path = f"{model_type}-encoder.axmodel"
    decoder_main_path = f"{model_type}-decoder-main.axmodel"
    decoder_loop_path = f"{model_type}-decoder-loop.axmodel"
    pe_path = f"{model_type}-positional_embedding.bin"
    token_path = f"{model_type}-tokens.txt"

    required_files = [os.path.join(model_path, i) for i in (encoder_path, decoder_main_path, decoder_loop_path, pe_path, token_path)]
    # Check file existence
    for i, file_path in enumerate(required_files):
        assert os.path.exists(file_path), f"{file_path} NOT exist"

    # Load encoder
    encoder = InferenceSession.load_from_model(required_files[0])
    # Load decoder main
    decoder_main = InferenceSession.load_from_model(required_files[1])
    # Load decoder loop
    decoder_loop = InferenceSession.load_from_model(required_files[2])
    # Load position embedding
    pe = np.fromfile(required_files[3], dtype=np.float32)
    # Load tokens
    tokens = []
    with open(required_files[4], "r") as f:
        for line in f:
            line = line.strip()
            tokens.append(line.split(" ")[0])

    return encoder, decoder_main, decoder_loop, pe, tokens


def compute_feature(wav_path, n_mels = WHISPER_N_MELS, padding = 480000):
    audio, sr = load_audio(wav_path)

    audio = np.concatenate((audio, np.zeros((padding,), dtype=np.float32)), axis=-1)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=WHISPER_N_FFT, hop_length=WHISPER_HOP_LENGTH, window="hann", center=True, pad_mode="reflect", power=2.0, n_mels=n_mels)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0

    # We pad 1500 frames at the end so that it is able to detect eot
    # You can use another value instead of 1500.
    # mel = np.concatenate((mel, np.zeros((n_mels, 1500), dtype=np.float32)), axis=-1)

    target = 3000
    if mel.shape[1] > target:
        # -50 so that there are some zero tail paddings.
        mel = mel[:, : target]
        mel[:, -50:] = 0

    # We don't need to pad it to 30 seconds now!
    if mel.shape[1] < target:
        mel = np.concatenate((mel, np.zeros((n_mels, target - mel.shape[1]), dtype=np.float32)), axis=-1)

    return mel


def supress_tokens(logits, is_initial):
    if is_initial:
        logits[WHISPER_EOT] = NEG_INF
        logits[WHISPER_BLANK] = NEG_INF

    logits[WHISPER_NO_TIMESTAMPS] = NEG_INF
    logits[WHISPER_SOT] = NEG_INF
    logits[WHISPER_NO_SPEECH] = NEG_INF
    logits[WHISPER_TRANSLATE] = NEG_INF
    return logits


class Whisper:
    def __init__(self, model_path, model_type, language="zh"):
        self.model_type = model_type
        self.encoder, self.decoder_main, self.decoder_loop, self.pe, self.tokens = load_models(model_path, model_type)
        self.WHISPER_N_TEXT_STATE = WHISPER_N_TEXT_STATE_MAP[self.model_type]
        self.language = language
        self.choose_language(self.language)

    def choose_language(self, lang):
        if lang not in WHISPER_LANGUAGES.keys():
            raise Exception(f"Unknown language: {lang}. Check languages.py for correct options.")
        SOT_SEQUENCE[1] = WHISPER_SOT + 1 + tuple(WHISPER_LANGUAGES.keys()).index(lang)

    def transcribe(self, input_audio) -> str:
        mel = compute_feature(input_audio, n_mels=WHISPER_N_MELS)
        x = self.encoder.run(input_feed={"mel": mel})
        n_layer_cross_k, n_layer_cross_v = x["n_layer_cross_k"], x["n_layer_cross_v"]
        x = self.decoder_main.run(input_feed={
            "tokens": SOT_SEQUENCE,
            "n_layer_cross_k": n_layer_cross_k,
            "n_layer_cross_v": n_layer_cross_v
        })
        logits, n_layer_self_k_cache, n_layer_self_v_cache = x["logits"], x["out_n_layer_self_k_cache"], x["out_n_layer_self_v_cache"]
        # Decode token
        logits = logits[0, -1, :]
        logits = supress_tokens(logits, is_initial=True)
        max_token_id = np.argmax(logits)
        output_tokens = []
        # Position embedding offset
        offset = SOT_SEQUENCE.shape[0]

        # Autoregressively run decoder until token meets EOT
        for i in range(WHISPER_N_TEXT_CTX - SOT_SEQUENCE.shape[0]):
            if max_token_id == WHISPER_EOT:
                break

            output_tokens.append(max_token_id)

            mask = np.zeros((WHISPER_N_TEXT_CTX,), dtype=np.float32)
            mask[: WHISPER_N_TEXT_CTX - offset - 1] = NEG_INF

            # Run decoder_loop
            x = self.decoder_loop.run(input_feed={
                "tokens": np.array([output_tokens[-1]], dtype=np.int32),
                "in_n_layer_self_k_cache": n_layer_self_k_cache,
                "in_n_layer_self_v_cache": n_layer_self_v_cache,
                "n_layer_cross_k": n_layer_cross_k,
                "n_layer_cross_v": n_layer_cross_v,
                "positional_embedding": self.pe[offset * self.WHISPER_N_TEXT_STATE : (offset + 1) * self.WHISPER_N_TEXT_STATE],
                "mask": mask
            })
            logits, n_layer_self_k_cache, n_layer_self_v_cache = x["logits"], x["out_n_layer_self_k_cache"], x["out_n_layer_self_v_cache"]

            # Decode token
            offset += 1
            logits = supress_tokens(logits.flatten(), is_initial=False)
            max_token_id = np.argmax(logits)
        s = b""
        for i in output_tokens:
            s += base64.b64decode(self.tokens[i])
        pd = s.decode().strip()
        if self.language == "zh":
            pd = zhconv.convert(pd, 'zh-hans')
        return pd

if __name__ == "__main__":
    model = Whisper(model_path="./models", model_type="tiny")
    print(model.transcribe("./demo.wav"))