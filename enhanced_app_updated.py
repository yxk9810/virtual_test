import gradio as gr
import subprocess
import os
import time
import threading
import gc
import shutil
from gemini_chat import GeminiChatSession
from system_prompt import prompt

class VirtualChatApp:
    def __init__(self):
        self.chat_session = GeminiChatSession(
            model="gemini-2.0-flash",
            system_instruction=prompt
        )
        self.chat_history = []
        # 使用welcome_video.mp4作为基础视频
        self.base_video_path = "welcome_video.mp4"
        self.current_video = None  # 当前显示的视频
        
        # 检查视频文件是否存在
        if not os.path.exists(self.base_video_path):
            print(f"警告: 基础视频文件不存在: {self.base_video_path}")
            print("请确保welcome_video.mp4文件存在于当前目录")
        else:
            print(f"基础视频文件已找到: {self.base_video_path}")
        
        # 创建初始欢迎视频
        self.initial_video = self.create_initial_video()
        self.current_video = self.initial_video
        
    def create_initial_video(self):
        """创建初始欢迎视频"""
        try:
            initial_video_path = "welcome_video.mp4"
            if os.path.exists(self.base_video_path):
                # 如果welcome_video.mp4已经存在，直接使用
                print(f"使用现有的欢迎视频: {initial_video_path}")
                return initial_video_path
            else:
                print("无法创建初始视频，welcome_video.mp4文件不存在")
                return None
        except Exception as e:
            print(f"创建初始视频失败: {e}")
            return None
        
    def process_message(self, user_input, history, progress=gr.Progress()):
        """处理用户消息的完整流程"""
        if not user_input.strip():
            return history, "", self.current_video, "⚠️ Please enter a message."
            
        # 更新状态
        progress(0.1, desc="🤖 Generating AI response...")
        
        # 1. 调用Gemini生成回复文本
        print(f"用户输入: {user_input}")
        ai_response = self.chat_session.send_message(user_input)
        
        if not ai_response:
            ai_response = "Sorry, I couldn't generate a response. Please try again."
        
        print(f"AI回复: {ai_response}")
        
        # 先将文本回复添加到历史记录（使用正确的格式）
        new_history = history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": ai_response}]
        
        # 2. 生成音频
        progress(0.3, desc="🎵 Generating audio...")
        audio_filename = f"audio_{int(time.time())}.wav"
        
        try:
            # 调用generate_audio.py生成音频
            cmd = [
                "python", "generate_audio.py", 
                f'"{ai_response}"',  # 使用AI回复作为文本
                audio_filename
            ]
            
            print(f"执行音频生成命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"音频生成成功: {audio_filename}")
                print(f"音频生成输出: {result.stdout}")
            else:
                print(f"音频生成失败: {result.stderr}")
                return new_history, "", self.current_video, f"⚠️ Audio generation failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            print("音频生成超时")
            return new_history, "", self.current_video, "⚠️ Audio generation timeout"
        except Exception as e:
            print(f"音频生成异常: {str(e)}")
            return new_history, "", self.current_video, f"⚠️ Audio generation error: {str(e)}"
        
        # 3. 生成视频
        progress(0.6, desc="🎬 Generating response video...")
        video_filename = f"video_{int(time.time())}.mp4"
        
        try:
            # 使用绝对路径调用run_latentsync.py生成视频
            current_dir = os.getcwd()
            welcome_video_abs = os.path.join(current_dir, "welcome_video.mp4")
            audio_file_abs = os.path.join(current_dir, audio_filename)
            video_file_abs = os.path.join(current_dir, video_filename)
            
            cmd = [
                "python", "run_latentsync.py",
                welcome_video_abs,  # 使用绝对路径
                audio_file_abs,     # 使用绝对路径
                video_file_abs      # 使用绝对路径
            ]
            
            print(f"执行视频生成命令: {' '.join(cmd)}")
            print(f"当前工作目录: {current_dir}")
            print(f"输入视频路径: {welcome_video_abs}")
            print(f"输入音频路径: {audio_file_abs}")
            print(f"输出视频路径: {video_file_abs}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
            
            print(f"视频生成返回码: {result.returncode}")
            print(f"视频生成标准输出: {result.stdout}")
            print(f"视频生成错误输出: {result.stderr}")
            
            if result.returncode == 0:
                # 检查输出文件是否真的存在
                if os.path.exists(video_file_abs) and os.path.getsize(video_file_abs) > 0:
                    print(f"视频生成成功: {video_filename}")
                    # 更新当前视频
                    self.current_video = video_filename
                    # 清理音频文件（保留视频文件）
                    self.cleanup_file(audio_filename)
                    return new_history, "", video_filename, f"✅ Successfully generated response video: {video_filename}"
                else:
                    print(f"视频文件不存在或为空: {video_file_abs}")
                    self.cleanup_file(audio_filename)
                    return new_history, "", self.current_video, "⚠️ Video file was not created properly"
            else:
                print(f"视频生成失败: {result.stderr}")
                # 如果视频生成失败，清理音频文件，但不更新视频
                self.cleanup_file(audio_filename)
                return new_history, "", self.current_video, f"⚠️ Video generation failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            print("视频生成超时")
            self.cleanup_file(audio_filename)
            return new_history, "", self.current_video, "⚠️ Video generation timeout"
        except Exception as e:
            print(f"视频生成异常: {str(e)}")
            self.cleanup_file(audio_filename)
            return new_history, "", self.current_video, f"⚠️ Video generation error: {str(e)}"
    
    def cleanup_file(self, filename):
        """清理临时文件"""
        try:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"已清理文件: {filename}")
        except Exception as e:
            print(f"清理文件失败 {filename}: {str(e)}")
    
    def clear_chat(self):
        """清空聊天历史"""
        self.chat_session.clear_history()
        # 重置为初始视频
        self.current_video = self.initial_video
        return [], "", self.current_video, "💬 Chat cleared! Ready for new conversation."

def create_interface():
    app = VirtualChatApp()
    
    # 自定义CSS样式
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .chat-container {
        height: 500px !important;
    }
    .video-container {
        height: 400px !important;
    }
    """
    
    with gr.Blocks(title="🤖 Virtual Chat AI", theme=gr.themes.Soft(), css=custom_css) as demo:
        # 标题和描述
        gr.Markdown("""
        # �� Virtual Chat with AI Host
        Chat with an AI virtual host! The AI will respond with both text and personalized video messages.
        """)
        
        with gr.Row():
            # 左侧：聊天区域
            with gr.Column(scale=3):
                # 聊天历史显示
                chatbot = gr.Chatbot(
                    label="💬 Chat History",
                    height=500,
                    show_label=True,
                    show_share_button=False,
                    show_copy_button=True,
                    elem_classes=["chat-container"],
                    type="messages"  # 使用messages格式
                )
                
                # 用户输入区域
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here and press Enter...",
                        label="Your Message",
                        lines=2,
                        scale=4,
                        max_lines=5
                    )
                    send_btn = gr.Button("Send 📤", variant="primary", scale=1, size="lg")
                
                # 控制按钮
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary", size="sm")
                    
            # 右侧：视频和状态区域
            with gr.Column(scale=2):
                # 视频显示区域
                video_output = gr.Video(
                    label="🎬 AI Response Video",
                    height=400,
                    autoplay=True,
                    show_share_button=False,
                    elem_classes=["video-container"],
                    value=app.initial_video  # 设置初始视频
                )
                
                # 视频控制按钮
                with gr.Row():
                    play_btn = gr.Button("▶️ Play Video", variant="primary", size="sm")
                    pause_btn = gr.Button("⏸️ Pause", variant="secondary", size="sm")
                    stop_btn = gr.Button("⏹️ Stop", variant="secondary", size="sm")
                
                # 状态显示
                status_text = gr.Textbox(
                    label="📊 Status",
                    value="🟢 Ready to chat! Type your message and press Send.",
                    interactive=False,
                    lines=3,
                    max_lines=5
                )
                
                # 系统信息
                gr.Markdown("""
                ### ℹ️ System Info
                - **Model**: Gemini 2.0 Flash
                - **TTS**: ChatterboxTTS
                - **Video**: LatentSync Pipeline
                - **GPU**: Auto-managed
                """)
        
        # 事件处理函数
        def handle_send(user_input, history):
            if not user_input.strip():
                return history, "", app.current_video, "⚠️ Please enter a message."
            
            # 调用处理函数
            new_history, cleared_input, video_file, final_status = app.process_message(user_input, history)
            
            return new_history, cleared_input, video_file, final_status
        
        def handle_clear():
            new_history, cleared_input, cleared_video, status = app.clear_chat()
            return new_history, cleared_input, cleared_video, status
        
        # 视频控制函数
        def play_video():
            return "Playing video..."
        
        def pause_video():
            return "Video paused"
        
        def stop_video():
            return "Video stopped"
        
        # 绑定事件
        send_btn.click(
            fn=handle_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, video_output, status_text],
            show_progress=True
        )
        
        msg_input.submit(
            fn=handle_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, video_output, status_text],
            show_progress=True
        )
        
        clear_btn.click(
            fn=handle_clear,
            outputs=[chatbot, msg_input, video_output, status_text]
        )
        
        # 视频控制按钮
        play_btn.click(fn=play_video, outputs=status_text)
        pause_btn.click(fn=pause_video, outputs=status_text)
        stop_btn.click(fn=stop_video, outputs=status_text)
        
        # 使用说明
        with gr.Accordion("📋 Usage Instructions", open=True):
            gr.Markdown("""
            ### Features:
            1. **Welcome Video** - Initial video loads automatically
            2. **Chat with AI** - Full Gemini conversation
            3. **Audio Generation** - Real TTS using ChatterboxTTS
            4. **Video Generation** - Real lipsync using LatentSync
            5. **Video Controls** - Manual play/pause/stop buttons
            6. **Progress Tracking** - Shows processing steps
            7. **Smart Video Update** - Only updates video when generation succeeds
            
            ### How it Works:
            - The welcome video loads when you open the page
            - When you send a message, AI generates a text response
            - Text is converted to speech using ChatterboxTTS
            - Video is generated using LatentSync pipeline
            - New video only replaces the old one when generation succeeds
            
            ### Processing Pipeline:
            1. **Text Generation** - Gemini AI creates response
            2. **Audio Generation** - ChatterboxTTS converts text to speech
            3. **Video Generation** - LatentSync creates lipsync video
            4. **Video Display** - New video replaces previous one (only if successful)
            
            ### Requirements:
            - **GPU**: Required for TTS and video generation
            - **LatentSync**: Must be cloned and configured
            - **Dependencies**: All required packages installed
            - **welcome_video.mp4**: Must exist in current directory
            """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True,
        favicon_path=None
    )
