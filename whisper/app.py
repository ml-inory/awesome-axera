import gradio as gr
from whisper_onnx import Whisper

model = Whisper(model_path="./models", model_type="small")

def transcribe(model_type, input_audio):
    model.load_model(model_type)
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
        # 模型类型
        model_type = gr.Dropdown(["tiny", "small"], value="small", label="Model type")
        # 上传音频
        audio = gr.Audio(label="Input audio", type="filepath", sources=["upload", "microphone"], format="wav")
        # 识别结果
        result = gr.Textbox(label="Result", interactive=False)
        # 识别按钮
        run_btn = gr.Button("Run")
        run_btn.click(fn=transcribe, inputs=[model_type, audio], outputs=[result])

demo.launch(server_name="0.0.0.0")