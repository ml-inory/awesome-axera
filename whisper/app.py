import gradio as gr
from ax_whisper import Whisper

model = Whisper(model_path="./models", model_type="small")

def transcribe(input_audio):
    return model.transcribe(input_audio)

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Whisper
        General-purpose speech recognition model from OpenAI.  
        Deployed on AX650N, visit https://axera-tech.com for product introduction.
        """
    )
    with gr.Group():
        # 上传音频
        audio = gr.Audio(label="Input audio", type="filepath", sources=["upload", "microphone"], format="wav")
        # 识别结果
        result = gr.Textbox(label="Result", interactive=False)
        # 识别按钮
        run_btn = gr.Button("Run")
        run_btn.click(fn=transcribe, inputs=[audio], outputs=[result])

demo.launch(server_name="0.0.0.0", server_port=8080, ssl_verify=False)