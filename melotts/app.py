import gradio as gr
from melotts_onnx import MeloTTS
import soundfile

model = MeloTTS(model_path="./models", language="ZH", speed=0.8)

def run(sentence):
    sr, audio = model.tts(sentence)
    return sr, audio

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # MeloTTS
        High-quality multi-lingual text-to-speech library by MyShell.ai. Support English, Spanish, French, Chinese, Japanese and Korean.  
        Deployed on AX650N, visit https://axera-tech.com for product introduction.
        """
    )
    # 输入句子
    sentence = gr.Textbox(label="Sentence")
    # 音频
    audio = gr.Audio(label="Output audio", type="numpy", format="wav")
    # 识别按钮
    run_btn = gr.Button("Run")
    run_btn.click(fn=run, inputs=[sentence], outputs=[audio])


demo.launch(server_name="0.0.0.0", server_port=8080, ssl_verify=False, ssl_certfile="../cert.pem", ssl_keyfile="../key.pem")