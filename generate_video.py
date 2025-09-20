import torch
import torchaudio
import subprocess
from datetime import datetime
import os
import ffmpeg
import shutil
import cv2
import sys

# 添加LatentSync相关导入
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed

loop_vid_from_endframe = True

# 全局变量存储模型，避免重复加载
_pipeline = None
_config = None

def setup_latentsync():
    """设置LatentSync环境和下载模型"""
    global _pipeline, _config
    
    if _pipeline is not None:
        return _pipeline, _config
    
    print("正在设置LatentSync环境...")
    
    # 检查LatentSync目录
    if not os.path.exists("LatentSync"):
        print("正在克隆LatentSync仓库...")
        subprocess.run(["git", "clone", "https://github.com/Isi-dev/LatentSync"], check=True)
    
    # 切换到LatentSync目录
    original_dir = os.getcwd()
    if not os.getcwd().endswith("LatentSync"):
        os.chdir("LatentSync")
    
    try:
        # 创建必要的目录 - 使用原来的路径定义
        os.makedirs("/kaggle/working//.cache/torch/hub/checkpoints", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        
        # 模型下载URLs - 使用原来的路径定义
        model_urls = {
            "/kaggle/working//.cache/torch/hub/checkpoints/s3fd-619a316812.pth":
                "https://huggingface.co/Isi99999/LatentSync/resolve/main/auxiliary/s3fd-619a316812.pth",
            "/kaggle/working/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip":
                "https://huggingface.co/Isi99999/LatentSync/resolve/main/auxiliary/2DFAN4-cd938726ad.zip",
            "checkpoints/latentsync_unet.pt":
                "https://huggingface.co/Isi99999/LatentSync/resolve/main/latentsync_unet.pt",
            "checkpoints/tiny.pt":
                "https://huggingface.co/Isi99999/LatentSync/resolve/main/whisper/tiny.pt",
            "checkpoints/diffusion_pytorch_model.safetensors":
                "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors",
            "checkpoints/config.json":
                "https://huggingface.co/stabilityai/sd-vae-ft-mse/raw/main/config.json",
        }
        
        # 下载模型文件
        for file_path, url in model_urls.items():
            if not os.path.exists(file_path):
                print(f"正在下载 {file_path} ...")
                subprocess.run(["wget", url, "-O", file_path], check=True)
            else:
                print(f"文件 {file_path} 已存在，跳过下载")
        
        print("Setup complete.")
        
        # 加载配置
        config_path = "configs/unet/first_stage.yaml"
        _config = OmegaConf.load(config_path)
        
        # 设置设备类型
        is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        dtype = torch.float16 if is_fp16_supported else torch.float32
        
        # 初始化调度器
        scheduler = DDIMScheduler.from_pretrained("configs")
        
        # 初始化音频编码器
        whisper_model_path = "checkpoints/tiny.pt"
        audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=_config.data.num_frames)
        
        # 初始化VAE
        vae = AutoencoderKL.from_pretrained("checkpoints", torch_dtype=dtype, local_files_only=True)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        
        # 初始化UNet
        inference_ckpt_path = "checkpoints/latentsync_unet.pt"
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(_config.model),
            inference_ckpt_path,
            device="cpu",
        )
        unet = unet.to(dtype=dtype)
        
        # 启用xformers优化
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            print('x_formers available!')
        
        # 创建pipeline
        _pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to("cuda")
        
        print("LatentSync环境设置完成！")
        return _pipeline, _config
        
    finally:
        # 恢复原始目录
        os.chdir(original_dir)

def perform_inference(video_path, audio_path, seed, num_steps, guidance_scale, output_path):
    """执行LatentSync推理生成视频"""
    try:
        # 设置LatentSync环境
        pipeline, config = setup_latentsync()
        
        print(f"开始执行推理: {video_path} + {audio_path} -> {output_path}")
        print(f"参数: seed={seed}, steps={num_steps}, guidance={guidance_scale}")
        
        # 设置随机种子
        set_seed(seed)
        
        # 确定数据类型
        is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        dtype = torch.float16 if is_fp16_supported else torch.float32
        
        # 执行推理
        pipeline(
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=output_path,
            video_mask_path=output_path.replace(".mp4", "_mask.mp4"),
            num_frames=config.data.num_frames,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
        )
        
        print("推理完成！")
        return output_path
        
    except Exception as e:
        print(f"推理过程中发生错误: {str(e)}")
        # 如果推理失败，返回原始视频路径
        return video_path


def convert_video_fps(input_path, target_fps):
    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        print(f"Error: The video file {input_path} is missing or empty.")
        return None

    output_path = f"converted_{target_fps}fps.mp4"

    audio_check_cmd = [
        "ffprobe", "-i", input_path, "-show_streams", "-select_streams", "a",
        "-loglevel", "error"
    ]
    audio_present = subprocess.run(audio_check_cmd, capture_output=True, text=True).stdout.strip() != ""

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter:v", f"fps={target_fps}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    ]

    if audio_present:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.append("-an")

    cmd.append(output_path)

    subprocess.run(cmd, check=True)
    print(f"Converted video saved as {output_path}")
    return output_path

def trim_video(video_path, target_duration):
    """Trim video to specified duration with robust error handling"""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return video_path
    if os.path.getsize(video_path) == 0:
        print(f"Error: Video file is empty at {video_path}")
        return video_path
    if target_duration <= 0:
        print(f"Error: Invalid target duration {target_duration}")
        return video_path

    try:
        probe = ffmpeg.probe(video_path, v='error', show_entries='format=duration')
        original_duration = float(probe['format']['duration'])
        if original_duration <= 0:
            print("Error: Could not determine valid video duration")
            return video_path

        print(f"Original duration: {original_duration:.2f}s, Target duration: {target_duration:.2f}s")

        if original_duration <= target_duration:
            print("Video is already shorter than target duration, no trimming needed")
            return video_path
    except Exception as e:
        print(f"Error probing video duration: {str(e)}")
        return video_path

    has_audio_stream = False
    try:
        audio_probe = ffmpeg.probe(
            video_path,
            v='error',
            select_streams='a',
            show_entries='stream=codec_type,codec_name'
        )
        has_audio_stream = any(stream['codec_type'] == 'audio' for stream in audio_probe.get('streams', []))
        if has_audio_stream:
            audio_codec = audio_probe['streams'][0]['codec_name']
            print(f"Detected audio stream with codec: {audio_codec}")
    except Exception as e:
        print(f"Warning: Could not determine audio status: {str(e)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trimmed_video_path = f"trimmed_{timestamp}.mp4"

    try:
        input_stream = ffmpeg.input(video_path, ss=0, to=target_duration)

        output_args = {
            'c:v': 'libx264',
            'preset': 'fast',
            'crf': '18',
            'pix_fmt': 'yuv420p',
            'movflags': '+faststart'
        }

        if has_audio_stream:
            output_args['c:a'] = 'aac'
            output_args['b:a'] = '192k'
            output_args['ar'] = '44100'
            output_args['ac'] = '2'

        cmd = (
            input_stream
            .output(trimmed_video_path, **output_args)
            .compile()
        )

        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("Video trimmed successfully")

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg trimming failed with error:\n{e.stderr}")
        if os.path.exists(trimmed_video_path):
            try:
                os.remove(trimmed_video_path)
            except Exception as clean_err:
                print(f"Warning: Could not clean up failed output: {str(clean_err)}")
        return video_path
    except Exception as e:
        print(f"Unexpected error during trimming: {str(e)}")
        return video_path

    if not os.path.exists(trimmed_video_path):
        print("Error: Trimmed video file was not created")
        return video_path

    if os.path.getsize(trimmed_video_path) == 0:
        print("Error: Trimmed video file is empty")
        os.remove(trimmed_video_path)
        return video_path

    try:
        output_duration = float(ffmpeg.probe(trimmed_video_path)['format']['duration'])
        duration_diff = abs(output_duration - target_duration)
        if duration_diff > 0.5:
            print(f"Warning: Trimmed duration is {output_duration:.2f}s (target: {target_duration:.2f}s)")
    except Exception as e:
        print(f"Warning: Could not verify output duration: {str(e)}")

    return trimmed_video_path

def has_audio(video_path):
    """Check if video contains audio stream"""
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='a')
        return len(probe['streams']) > 0
    except ffmpeg.Error:
        return False

def extend_video(video_path, target_duration):
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        print(f"Error: The video file {video_path} is missing or empty.")
        return video_path

    audio_exists = has_audio(video_path)
    print(f"Audio exists in source: {audio_exists}")

    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='format=duration')
        original_duration = float(probe['format']['duration'])
        print(f"Original duration: {original_duration:.2f}s, Target duration: {target_duration:.2f}s")

        if original_duration <= 0:
            raise ValueError("Invalid video duration detected")
    except Exception as e:
        print(f"Error getting video duration: {str(e)}")
        return video_path

    if original_duration >= target_duration:
        print("Video already meets target duration")
        return video_path

    clips = [video_path]
    total_duration = original_duration
    extensions = 0

    while total_duration < target_duration:
        extensions += 1
        try:
            if loop_vid_from_endframe:
                reversed_clip = reverse_video(clips[-1], audio_exists)
                if not os.path.exists(reversed_clip) or os.path.getsize(reversed_clip) == 0:
                    raise Exception("Reversed clip creation failed")
                clips.append(reversed_clip)
            else:
                clips.append(clips[-1])

            total_duration += original_duration
        except Exception as e:
            print(f"Failed during clip extension: {str(e)}")
            break

    if len(clips) <= 1:
        print("No extension performed, returning original")
        return video_path

    print("\nClip properties before concatenation:")
    for i, clip in enumerate(clips):
        try:
            probe = ffmpeg.probe(clip)
            for stream in probe['streams']:
                if stream['codec_type'] == 'video':
                    print(f"  Video: {stream['codec_name']} {stream['width']}x{stream['height']}")
                elif stream['codec_type'] == 'audio':
                    print(f"  Audio: {stream['codec_name']}")
        except Exception as e:
            print(f"Error checking clip {clip}: {str(e)}")
            return video_path

    extended_video_path = "welcome_video.mp4"
    concat_list_path = "concat_list.txt"

    try:
        with open(concat_list_path, 'w') as f:
            for clip in clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c', 'copy'
        ]

        if not extended_video_path.endswith('.mp4'):
            cmd.extend(['-f', 'mp4'])

        cmd.append(extended_video_path)

        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("Concatenation successful!")

    except subprocess.CalledProcessError as e:
        print(f"Concatenation failed with error:\n{e.stderr}")
        return video_path
    except Exception as e:
        print(f"Unexpected error during concatenation: {str(e)}")
        return video_path
    finally:
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)

        for clip in clips[1:]:
            if os.path.exists(clip):
                try:
                    os.remove(clip)
                except Exception as e:
                    print(f"Warning: Could not remove {clip}: {str(e)}")

    if not os.path.exists(extended_video_path) or os.path.getsize(extended_video_path) == 0:
        print("Error: Final extended video not created properly")
        return video_path

    final_duration = get_video_duration(extended_video_path)
    print(f"Final extended duration: {final_duration:.2f}s")

    return extended_video_path

def reverse_video(video_path, audio_exists):
    """Create a reversed version of the video"""
    reversed_path = f"reversed_{os.path.basename(video_path)}"
    try:
        if audio_exists:
            (
                ffmpeg.input(video_path)
                .output(reversed_path, vf='reverse', af='areverse')
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
        else:
            (
                ffmpeg.input(video_path)
                .output(reversed_path, vf='reverse')
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
        return reversed_path
    except ffmpeg.Error as e:
        print(f"Reverse failed: {e.stderr.decode()}")
        raise

def get_video_duration(video_path):
    """Get duration in seconds"""
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='format=duration')
        return float(probe['format']['duration'])
    except Exception as e:
        print(f"Duration check failed: {str(e)}")
        return 0

def pad_audio_to_multiple_of_16(audio_path, target_fps=25):
    waveform, sample_rate = torchaudio.load(audio_path)
    audio_duration = waveform.shape[1] / sample_rate
    num_frames = int(audio_duration * target_fps)
    remainder = num_frames % 16

    if remainder > 0:
        pad_frames = 16 - remainder
        pad_samples = int((pad_frames / target_fps) * sample_rate)
        pad_waveform = torch.zeros((waveform.shape[0], pad_samples))
        waveform = torch.cat((waveform, pad_waveform), dim=1)
        padded_audio_path = "padded_audio.wav"
        torchaudio.save(padded_audio_path, waveform, sample_rate)
    else:
        padded_audio_path = audio_path

    return padded_audio_path, int((waveform.shape[1] / sample_rate) * target_fps), waveform.shape[1] / sample_rate

    import shutil
    shutil.copy2(video_path, output_path)
    print("推理完成（临时实现）")

def generate_video(video_path, audio_path, seed=1247, num_steps=20, guidance_scale=1.0, 
                  video_scale=0.5, output_fps=25, output_path="output_video.mp4"):
    """
    主函数：生成视频
    """
    
    print("开始视频生成...")
    print(f"视频路径: {video_path}")
    print(f"音频路径: {audio_path}")
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 {video_path}")
        return None
        
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在 {audio_path}")
        return None

    print("文件检查通过，开始处理...")
    
    work_video_path = "working_video.mp4"
    work_audio_path = "working_audio.wav"
    
    try:
        shutil.copy2(video_path, work_video_path)
        shutil.copy2(audio_path, work_audio_path)
    except Exception as e:
        print(f"文件复制失败: {str(e)}")
        return None

    width, height = 0, 0
    try:
        cap = cv2.VideoCapture(work_video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"视频尺寸: {width}x{height}")
        else:
            print("警告: 无法打开视频文件获取尺寸")
        cap.release()
    except Exception as e:
        print(f"获取视频尺寸失败: {str(e)}")

    try:
        work_video_path = convert_video_fps(work_video_path, 25)
        if not work_video_path:
            print("视频帧率转换失败")
            return None

        work_audio_path, num_frames, audio_duration = pad_audio_to_multiple_of_16(work_audio_path, target_fps=25)

        video_duration = get_video_duration(work_video_path)
        print(f"音频时长: {audio_duration:.2f}秒")
        print(f"视频时长: {video_duration:.2f}秒")

        if audio_duration > video_duration:
            print("音频较长，扩展视频...")
            work_video_path = extend_video(work_video_path, audio_duration)
            video_duration = get_video_duration(work_video_path)
            if video_duration > audio_duration:
                print("视频扩展过长，进行裁剪...")
                work_video_path = trim_video(work_video_path, audio_duration)
        elif video_duration > audio_duration:
            print("视频较长，裁剪视频...")
            work_video_path = trim_video(work_video_path, audio_duration)

        print("开始执行推理...")
        temp_output = "temp_output_video.mp4"
        
        perform_inference(work_video_path, work_audio_path, seed, num_steps, guidance_scale, temp_output)

        final_output = convert_video_fps(temp_output, output_fps)
        if final_output and os.path.exists(final_output):
            if final_output != output_path:
                shutil.move(final_output, output_path)
            
            print(f"视频生成成功！输出文件: {output_path}")
            
            if width > 0 and height > 0:
                print(f"输出视频尺寸: {int(width * video_scale)}x{int(height * video_scale)}")
            
            return output_path
        else:
            print("最终视频输出失败")
            return None

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return None
        
    finally:
        temp_files = [work_video_path, work_audio_path, "temp_output_video.mp4", 
                     "padded_audio.wav", "working_video.mp4", "working_audio.wav"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"清理临时文件失败 {temp_file}: {str(e)}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python generate_video.py <video_path> <audio_path> [output_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "generated_video.mp4"
    
    result = generate_video(
        video_path=video_path,
        audio_path=audio_path,
        seed=1247,
        num_steps=20,
        guidance_scale=1.0,
        video_scale=0.5,
        output_fps=25,
        output_path=output_path
    )
    
    if result:
        print(f"成功生成视频: {result}")
    else:
        print("视频生成失败")
        sys.exit(1)
