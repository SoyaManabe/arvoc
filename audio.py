from gtts import gTTS
import os

def text_to_speech(text, lang="ja", output_file="output.mp3"):
    """テキストを音声に変換して再生"""
    tts = gTTS(text=text, lang=lang)
    tts.save(output_file)
    os.system(f"open {output_file}")  # Windows 用（Mac: open, Linux: mpg321）


