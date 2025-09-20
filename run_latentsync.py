#!/usr/bin/env python3
"""
LatentSync视频生成外部脚本
自动管理目录切换，在LatentSync目录下执行generate_video.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def find_latentsync_directory():
    """查找LatentSync目录"""
    print("🔍 正在查找LatentSync目录...")
    
    # 检查当前目录
    if os.path.exists("LatentSync"):
        latentsync_path = os.path.abspath("LatentSync")
        print(f"✅ 在当前目录找到LatentSync: {latentsync_path}")
        return latentsync_path
    
    # 检查父目录
    parent_dir = os.path.dirname(os.getcwd())
    latentsync_in_parent = os.path.join(parent_dir, "LatentSync")
    if os.path.exists(latentsync_in_parent):
        print(f"✅ 在父目录找到LatentSync: {latentsync_in_parent}")
        return latentsync_in_parent
    
    # 向上查找3级目录
    current_dir = os.getcwd()
    for i in range(3):
        parent = os.path.dirname(current_dir)
        latentsync_path = os.path.join(parent, "LatentSync")
        if os.path.exists(latentsync_path):
            print(f"✅ 在上级目录找到LatentSync: {latentsync_path}")
            return latentsync_path
        current_dir = parent
    
    return None

def clone_latentsync():
    """克隆LatentSync仓库"""
    print("📥 正在克隆LatentSync仓库...")
    try:
        subprocess.run(["git", "clone", "https://github.com/Isi-dev/LatentSync"], check=True)
        latentsync_path = os.path.abspath("LatentSync")
        print(f"✅ LatentSync克隆完成: {latentsync_path}")
        return latentsync_path
    except subprocess.CalledProcessError as e:
        print(f"❌ 克隆失败: {e}")
        return None

def setup_latentsync():
    """设置LatentSync环境"""
    latentsync_path = find_latentsync_directory()
    
    if latentsync_path is None:
        print("📥 未找到LatentSync目录，正在克隆...")
        latentsync_path = clone_latentsync()
        if latentsync_path is None:
            return None
    
    # 验证LatentSync目录结构
    config_file = os.path.join(latentsync_path, "configs", "unet", "first_stage.yaml")
    if not os.path.exists(config_file):
        print(f"❌ 错误：未找到LatentSync配置文件: {config_file}")
        return None
    
    print(f"✅ LatentSync环境准备完成: {latentsync_path}")
    return latentsync_path

def copy_generate_video_script(latentsync_path):
    """复制generate_video.py到LatentSync目录"""
    current_script = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script)
    source_script = os.path.join(current_dir, "generate_video.py")
    target_script = os.path.join(latentsync_path, "generate_video.py")
    
    if os.path.exists(source_script):
        try:
            shutil.copy2(source_script, target_script)
            print(f"✅ 已复制generate_video.py到: {target_script}")
        except Exception as e:
            print(f"❌ 复制脚本失败: {e}")
            return None
    else:
        print(f"❌ 未找到源脚本: {source_script}")
        return None
    
    # 复制welcome_video.mp4文件
    welcome_video_source = os.path.join(current_dir, "welcome_video.mp4")
    welcome_video_target = os.path.join(latentsync_path, "welcome_video.mp4")
    if os.path.exists(welcome_video_source):
        try:
            shutil.copy2(welcome_video_source, welcome_video_target)
            print(f"✅ 已复制welcome_video.mp4到: {welcome_video_target}")
        except Exception as e:
            print(f"⚠️ 复制welcome_video.mp4失败: {e}")
    else:
        print(f"⚠️ 未找到welcome_video.mp4: {welcome_video_source}")
    
    return target_script

def run_latentsync_generation(video_path, audio_path, output_path=None):
    """在LatentSync目录下运行视频生成"""
    print("🚀 开始LatentSync视频生成...")
    
    # 设置LatentSync环境
    latentsync_path = setup_latentsync()
    if latentsync_path is None:
        return False
    
    # 复制generate_video.py脚本
    script_path = copy_generate_video_script(latentsync_path)
    if script_path is None:
        return False
    
    # 保存当前目录
    original_dir = os.getcwd()
    
    try:
        # 切换到LatentSync目录
        os.chdir(latentsync_path)
        print(f"📁 切换到工作目录: {os.getcwd()}")
        
        # 构建命令
        cmd = [sys.executable, "generate_video.py", video_path, audio_path]
        if output_path:
            cmd.append(output_path)
        
        print(f"🎬 执行命令: {' '.join(cmd)}")
        
        # 执行视频生成
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            print("✅ 视频生成成功！")
            return True
        else:
            print("❌ 视频生成失败")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 执行失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return False
    finally:
        # 恢复原始目录
        os.chdir(original_dir)
        print(f"📁 恢复原始目录: {os.getcwd()}")

def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("📖 用法: python run_latentsync.py <video_path> <audio_path> [output_path]")
        print("💡 提示: 此脚本会自动管理LatentSync目录切换")
        print("")
        print("示例:")
        print("  python run_latentsync.py input_video.mp4 input_audio.wav")
        print("  python run_latentsync.py input_video.mp4 input_audio.wav output_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # 检查输入文件
    if not os.path.exists(video_path):
        print(f"❌ 错误: 视频文件不存在 {video_path}")
        sys.exit(1)
    
    if not os.path.exists(audio_path):
        print(f"❌ 错误: 音频文件不存在 {audio_path}")
        sys.exit(1)
    
    print("🎬 LatentSync视频生成器")
    print(f"📹 视频文件: {video_path}")
    print(f"🎵 音频文件: {audio_path}")
    if output_path:
        print(f"📤 输出文件: {output_path}")
    print("")
    
    # 运行视频生成
    success = run_latentsync_generation(video_path, audio_path, output_path)
    
    if success:
        print("🎉 视频生成完成！")
        sys.exit(0)
    else:
        print("❌ 视频生成失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
