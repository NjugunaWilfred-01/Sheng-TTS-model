"""
Interactive Gradio Web UI for the Swahili & Sheng Speech-to-Speech (S2S) Pipeline.
Includes Multi-Turn Conversation Memory, Preset Live Demo Scenarios, and Model Controls.
"""
import os
from pathlib import Path
import gradio as gr
from pipeline import SpeechToSpeechPipeline
from config import VOICE_OPTIONS, DEFAULT_VOICE, GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, BASE_DIR
from sheng_lexicon import SHENG_DICTIONARY

# Initialize Pipeline
print("🚀 Initializing Swahili & Sheng S2S Pipeline...")
s2s_pipeline = SpeechToSpeechPipeline()

DEMO_SAMPLES = {
    "🌟 1. Greeting & Rada": str(BASE_DIR / "assets" / "demo_samples" / "demo_1_greeting.mp3"),
    "💼 2. Work & Hustle": str(BASE_DIR / "assets" / "demo_samples" / "demo_5_work.mp3"),
    "🍲 3. Lunch at Kibanda": str(BASE_DIR / "assets" / "demo_samples" / "demo_2_food.mp3"),
    "🚌 4. Nganya to Tao": str(BASE_DIR / "assets" / "demo_samples" / "demo_3_transport.mp3"),
    "🎉 5. Weekend Vibes": str(BASE_DIR / "assets" / "demo_samples" / "demo_6_weekend.mp3"),
    "👟 6. Luku & Drip": str(BASE_DIR / "assets" / "demo_samples" / "demo_4_drip.mp3")
}

BACKEND_OPTIONS = {
    "Instant Heuristic Sheng Brain (<2ms Latency)": "heuristic",
    "Fine-Tuned Sheng LoRA (Qwen2.5-Instruct)": "lora"
}


def process_voice_turn(audio_input, voice_choice, rate_pct, pitch_hz, backend_choice, history):
    """Handles a single conversational voice turn with multi-turn dialogue memory."""
    if history is None:
        history = []

    if not audio_input:
        return (
            "⚠️ Tafadhali rekodi au weka sauti kwanza (Please record or upload audio first).",
            "",
            "",
            None,
            "N/A",
            history
        )

    voice_id = VOICE_OPTIONS.get(voice_choice, DEFAULT_VOICE)
    backend_id = BACKEND_OPTIONS.get(backend_choice, "lora")
    rate_str = f"{'+' if rate_pct >= 0 else ''}{int(rate_pct)}%"
    pitch_str = f"{'+' if pitch_hz >= 0 else ''}{int(pitch_hz)}Hz"

    result = s2s_pipeline.process_audio(
        input_audio_path=audio_input,
        voice=voice_id,
        rate=rate_str,
        pitch=pitch_str,
        backend=backend_id
    )

    if not result.get("success"):
        return (
            f"❌ Hitilafu: {result.get('error', 'Unknown error')}",
            "",
            "",
            None,
            "N/A",
            history
        )

    user_raw = result["user_raw_transcription"]
    user_sheng = result["user_normalized_sheng"]
    bot_reply = result["bot_response_text"]
    audio_out = result["output_audio_path"]
    latencies = result["latencies"]

    # Append to multi-turn chat history (list of role/content message dicts for Gradio 6)
    history.append({"role": "user", "content": user_sheng or user_raw})
    history.append({"role": "assistant", "content": bot_reply})

    latency_str = (
        f"⚡ **Glass-to-Glass:** `{latencies['total_glass_to_glass_ms']} ms` | "
        f"🎙️ **ASR:** `{latencies['asr_ms']} ms` | "
        f"🧠 **LLM:** `{latencies['llm_ms']} ms` ({backend_id}) | "
        f"🔊 **TTS:** `{latencies['tts_ms']} ms`"
    )

    return (
        user_raw,
        user_sheng,
        bot_reply,
        audio_out,
        latency_str,
        history
    )


def reset_chat_history():
    """Clears both backend LLM history and UI chat history."""
    s2s_pipeline.reset_conversation()
    return "", "", "", None, "⚡ **Latency:** *Maongezi yamefutwa (History Cleared)*", []


def load_preset_scenario(scenario_key):
    """Loads a pre-recorded demo scenario audio file."""
    return DEMO_SAMPLES.get(scenario_key, None)


# Build Gradio UI
custom_css = """
.container { max-width: 950px; margin: auto; }
.header-box { text-align: center; margin-bottom: 20px; }
.metric-box { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px; margin-top: 10px; font-size: 14px; }
.demo-btn { margin-bottom: 5px; }
"""

with gr.Blocks(title="Swahili & Sheng S2S Agent", theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Column(elem_classes=["container"]):
        gr.Markdown(
            """
            # 🇰🇪 Swahili & Sheng Speech-to-Speech (S2S) Agent
            ### *Ongea Kiswahili au Sheng ya Nairobi upate jibu la sauti papo hapo!*
            Speak in Swahili or Nairobi Sheng and receive an authentic spoken response with low latency.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Ongea na Bot (Live Voice Recording)")
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Rekodi Sauti Yako (Record or Upload Audio)"
                )

                with gr.Row():
                    submit_btn = gr.Button("🚀 Ongea na Bot / Send Voice", variant="primary", size="lg")
                    reset_btn = gr.Button("🗑️ Reset Chat", variant="secondary", size="lg")

                with gr.Accordion("📁 Optional Pre-recorded Samples (For Testing)", open=False):
                    with gr.Row():
                        demo_btn1 = gr.Button("1. Greeting & Rada", size="sm")
                        demo_btn2 = gr.Button("2. Work & Hustle", size="sm")
                    with gr.Row():
                        demo_btn3 = gr.Button("3. Kibanda Lunch", size="sm")
                        demo_btn4 = gr.Button("4. Nganya to Tao", size="sm")
                    with gr.Row():
                        demo_btn5 = gr.Button("5. Weekend Vibes", size="sm")
                        demo_btn6 = gr.Button("6. Luku & Drip", size="sm")

                with gr.Accordion("⚙️ Voice & Brain Settings", open=False):
                    voice_dropdown = gr.Dropdown(
                        choices=list(VOICE_OPTIONS.keys()),
                        value=list(VOICE_OPTIONS.keys())[0],
                        label="Chagua Sauti ya Bot / Select Bot Voice"
                    )
                    backend_dropdown = gr.Dropdown(
                        choices=list(BACKEND_OPTIONS.keys()),
                        value=list(BACKEND_OPTIONS.keys())[0],
                        label="Chagua LLM Brain / Select LLM Backend"
                    )
                    with gr.Row():
                        rate_slider = gr.Slider(
                            minimum=-25,
                            maximum=25,
                            value=0,
                            step=1,
                            label="Kasi / Speed (%)"
                        )
                        pitch_slider = gr.Slider(
                            minimum=-20,
                            maximum=20,
                            value=0,
                            step=1,
                            label="Sauti / Pitch (Hz)"
                        )

            with gr.Column(scale=1):
                gr.Markdown("### 🔊 Jibu la Bot (Audio Output)")
                audio_output = gr.Audio(
                    label="Sauti ya Bot / Bot Spoken Response",
                    autoplay=True
                )
                latency_display = gr.Markdown("⚡ **Latency:** *Subiri sauti...*", elem_classes=["metric-box"])

                gr.Markdown("### 💬 Maongezi Yote (Multi-Turn Chat)")
                chatbot = gr.Chatbot(label="Conversation History", height=280)

        with gr.Accordion("📝 Maelezo ya Ziada (Live Transcripts)", open=False):
            with gr.Row():
                user_raw_box = gr.Textbox(label="1. Raw ASR Transcription (Whisper)", interactive=False)
                user_sheng_box = gr.Textbox(label="2. Normalized Sheng Slang", interactive=False)
            bot_reply_box = gr.Textbox(label="3. Bot Response Text (Sheng Persona)", interactive=False, lines=2)

        with gr.Accordion("📚 Sheng Slang Cheat Sheet (~50 Target Words)", open=False):
            lexicon_text = "\n".join([f"- **{k}**: {v}" for k, v in list(SHENG_DICTIONARY.items())[:30]])
            gr.Markdown(lexicon_text)

        # Event Handlers
        submit_btn.click(
            fn=process_voice_turn,
            inputs=[audio_input, voice_dropdown, rate_slider, pitch_slider, backend_dropdown, chatbot],
            outputs=[user_raw_box, user_sheng_box, bot_reply_box, audio_output, latency_display, chatbot]
        )

        reset_btn.click(
            fn=reset_chat_history,
            inputs=[],
            outputs=[user_raw_box, user_sheng_box, bot_reply_box, audio_output, latency_display, chatbot]
        )

        demo_btn1.click(lambda: load_preset_scenario("🌟 1. Greeting & Rada"), outputs=[audio_input])
        demo_btn2.click(lambda: load_preset_scenario("💼 2. Work & Hustle"), outputs=[audio_input])
        demo_btn3.click(lambda: load_preset_scenario("🍲 3. Lunch at Kibanda"), outputs=[audio_input])
        demo_btn4.click(lambda: load_preset_scenario("🚌 4. Nganya to Tao"), outputs=[audio_input])
        demo_btn5.click(lambda: load_preset_scenario("🎉 5. Weekend Vibes"), outputs=[audio_input])
        demo_btn6.click(lambda: load_preset_scenario("👟 6. Luku & Drip"), outputs=[audio_input])

if __name__ == "__main__":
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=False
    )
