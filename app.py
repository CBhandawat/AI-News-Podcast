import pkgutil

if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = pkgutil.zipimporter

import gradio as gr
import os
import soundfile as sf
from scraper import scrape_new_posts, generate_podcast_summary_from_file
from audible import (
    generate_audio_sarvam,
    generate_audio_elevenlabs,
)
from musetalk_wrapper import generate_talking_video  # Added missing import

# ---------------- News Fetch ---------------- #
def get_latest_news():
    try:
        scrape_new_posts()
        summary = generate_podcast_summary_from_file()
        return summary, "✅ News summary generated!"
    except Exception as e:
        return "", f"❌ Error fetching news: {e}"

# ---------------- Audio + Video Generation ---------------- #
def generate_audio(summary, voice_option):
    if not summary:
        return None, None, "❌ No summary available. Fetch news first."
    
    output_file = None
    try:
        # --- Generate audio ---
        if voice_option == "Sarvam":
            output_file = "podcast_sarvam.wav"
            generate_audio_sarvam(summary, output_file)
        elif voice_option == "ElevenLabs":
            output_file = "podcast_elevenlabs.wav"
            generate_audio_elevenlabs(summary, output_file)

        # --- Generate video with MuseTalk ---
        talking_video = None
        image_path = "host.jpg"
        if not os.path.exists(image_path):
            return output_file, None, f"✅ Audio generated with {voice_option}! ❌ Missing host.jpg for video."
        
        if output_file and os.path.exists(output_file):
            talking_video = generate_talking_video(
                audio_path=output_file,
                image_path=image_path,
                output_path="talking_news.mp4"
            )

        if output_file and os.path.exists(output_file):
            status_msg = f"✅ Audio generated with {voice_option}!"
            if talking_video and os.path.exists(talking_video):
                status_msg += " 🎥 Talking video created!"
            else:
                status_msg += " ❌ Video generation failed (check MuseTalk setup)."
            return output_file, talking_video, status_msg
        else:
            return None, None, "❌ Audio generation failed."
    except Exception as e:
        return None, None, f"❌ Error generating audio/video: {e}"

# ---------------- Gradio Interface ---------------- #
def create_interface():
    with gr.Blocks(title="AI News Podcast", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ AI News Daily Podcast with Talking Head")
        gr.Markdown("Get the latest AI news, generate audio with Sarvam or ElevenLabs, and create a talking video with MuseTalk.")
        
        # Step 1: Fetch latest news
        fetch_btn = gr.Button("📥 Get me the latest AI news", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        
        # Step 2: Show summary
        summary_text = gr.Textbox(
            label="📰 News Summary (Podcast-style)",
            lines=15,
            interactive=False,
            placeholder="Summary will appear here after fetching news."
        )
        
        # Step 3: Choose voice & generate audio+video
        with gr.Row():
            voice_option = gr.Radio(
                choices=["Sarvam", "ElevenLabs"],
                label="🎤 Choose a voice",
                value="Sarvam"
            )
            audio_btn = gr.Button("🔊 Generate Podcast Audio + 🎥 Talking Video", variant="secondary")
        
        audio_file = gr.Audio(label="Generated Audio", type="filepath")
        video_file = gr.Video(label="Talking Head Video")
        audio_status = gr.Textbox(label="Status", interactive=False)
        
        # Bind buttons
        fetch_btn.click(
            fn=get_latest_news,
            inputs=None,
            outputs=[summary_text, status]
        )
        
        audio_btn.click(
            fn=generate_audio,
            inputs=[summary_text, voice_option],
            outputs=[audio_file, video_file, audio_status]
        )
        
        gr.Markdown("👉 Video uses generated audio (Sarvam/ElevenLabs) with `host.jpg`.")
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861)