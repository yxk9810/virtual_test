import torch
import torchaudio
import subprocess
from datetime import datetime
import os
import ffmpeg
import shutil
import cv2
import sys

loop_vid_from_endframe = True

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

    extended_video_path = "extended_video.mp4"
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

def perform_inference(video_path, audio_path, seed, num_steps, guidance_scale, output_path):
    """执行推理生成视频 - 这里需要根据你的具体模型实现"""
    # 这是一个占位符函数，你需要根据你的具体模型来实现
    # 例如使用你的Grok listener或其他视频生成模型
    print(f"执行推理: {video_path} + {audio_path} -> {output_path}")
    print(f"参数: seed={seed}, steps={num_steps}, guidance={guidance_scale}")
    
    # 临时实现：直接复制输入视频作为输出
    # 在实际使用中，这里应该调用你的视频生成模型
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
