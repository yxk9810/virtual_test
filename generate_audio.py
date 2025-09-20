import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import sys
import os

def generate_audio(text, output_path="generated_audio.wav"):
    """生成音频文件"""
    try:
        # 初始化模型
        model = ChatterboxTTS.from_pretrained(device="cuda")
        device = 'cuda'
        
        # 音频提示路径
        AUDIO_PROMPT_PATH = "./demo1_audio.wav"
        
        # 生成音频
        wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
        
        # 保存音频文件
        ta.save(output_path, wav, model.sr)
        
        print(f"音频生成成功: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"音频生成失败: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python generate_audio.py <text> [output_path]")
        sys.exit(1)
    
    text = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "generated_audio.wav"
    
    result = generate_audio(text, output_path)
    if result:
        print(f"成功生成音频: {result}")
    else:
        print("音频生成失败")
        sys.exit(1)
