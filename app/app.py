"""Gradio web app for the Poultry Fecal-Image Screening system.

Run locally :  python app.py
Hugging Face:  push this folder as a Gradio Space (see README.md)
Env vars    :  PORT (default 7860), USE_CLIP=0 to disable the CLIP gate (lighter / faster start)
"""
import os

from poultry_core import DISCLAIMER, PoultryAnalyzer

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(HERE, "model_bundle")

analyzer = PoultryAnalyzer(BUNDLE_DIR)


def run(img):
    if img is None:
        return "Please upload an image first.", None, None, None
    res = analyzer.analyze(img)
    msg = analyzer.message(res)
    if analyzer.warnings:
        msg += "\n\n> Note: " + " ".join(sorted(set(analyzer.warnings)))
    show_probs = res["verdict"] in ("PREDICTION", "UNCERTAIN")
    probs = {analyzer.pretty(k): v for k, v in res["probs"].items()} if show_probs else None
    return msg, probs, res["cam"], res["info"]


def build_demo():
    import gradio as gr
    with gr.Blocks(title="Poultry Fecal-Image Screening") as demo:
        gr.Markdown(
            "# Poultry Disease Screening from Fecal Images\n"
            "Upload a close-up photo of chicken droppings. The system first checks that the "
            "image really shows poultry droppings (rejecting other animals and unrelated "
            "pictures), then screens for **Coccidiosis, Salmonellosis, Newcastle disease** "
            f"or **Healthy**.\n\n_{DISCLAIMER}_")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Fecal image", sources=["upload", "webcam"])
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                verdict = gr.Markdown()
                probs = gr.Label(num_top_classes=4, label="Class probabilities")
                cam = gr.Image(label="Grad-CAM: regions that influenced the prediction")
                details = gr.JSON(label="Gate diagnostics")
        btn.click(run, inputs=inp, outputs=[verdict, probs, cam, details])
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0",
                                   server_port=int(os.environ.get("PORT", 7860)))
