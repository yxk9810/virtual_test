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
        # 使用本地视频路径
        self.base_video_path = os.path.expanduser("~/Downloads/kling_20250806_Image_to_Video_this_woman_3742_0.mp4")
        
        # 检查视频文件是否存在
        if not os.path.exists(self.base_video_path):
            print(f"警告: 基础视频文件不存在: {self.base_video_path}")
        else:
            print(f"基础视频文件已找到: {self.base_video_path}")
        
        # 创建初始欢迎视频
        self.initial_video = self.create_initial_video()
        
    def create_initial_video(self):
        """创建初始欢迎视频"""
        try:
            initial_video_path = "welcome_video.mp4"
            if os.path.exists(self.base_video_path):
                shutil.copy2(self.base_video_path, initial_video_path)
                print(f"初始欢迎视频已创建: {initial_video_path}")
                return initial_video_path
            else:
                print("无法创建初始视频，基础视频文件不存在")
                return None
        except Exception as e:
            print(f"创建初始视频失败: {e}")
            return None
        
    def process_message(self, user_input, history, progress=gr.Progress()):
        """处理用户消息的完整流程"""
        if not user_input.strip():
            return history, "", None, "⚠️ Please enter a message."
            
        # 更新状态
        progress(0.1, desc="🤖 Generating AI response...")
        
        # 1. 调用Gemini生成回复文本
        print(f"用户输入: {user_input}")
        ai_response = self.chat_session.send_message(user_input)
        
        if not ai_response:
            ai_response = "Sorry, I couldn't generate a response. Please try again."
        
        print(f"AI回复: {ai_response}")
        
        # 先将文本回复添加到历史记录
        new_history = history + [[user_input, ai_response]]
        
        # 2. 测试音频生成
        progress(0.3, desc="🎵 Testing audio generation...")
        audio_filename = f"audio_{int(time.time())}.wav"
        
        # 创建测试音频文件（静音）
        try:
            import numpy as np
            import torch
            import torchaudio
            
            # 创建3秒的静音音频
            sample_rate = 22050
            duration = 3.0
            samples = int(sample_rate * duration)
            silent_audio = np.zeros(samples)
            
            # 保存为WAV文件
            torchaudio.save(audio_filename, torch.tensor(silent_audio).unsqueeze(0), sample_rate)
            print(f"测试音频文件已创建: {audio_filename}")
            
        except Exception as e:
            print(f"创建测试音频失败: {str(e)}")
            return new_history, "", None, f"⚠️ Audio creation failed: {str(e)}"
        
        # 3. 测试视频处理
        progress(0.6, desc="🎬 Generating response video...")
        video_filename = f"video_{int(time.time())}.mp4"
        
        try:
            # 简单的视频复制作为测试
            shutil.copy2(self.base_video_path, video_filename)
            print(f"响应视频文件已创建: {video_filename}")
            
        except Exception as e:
            print(f"创建响应视频失败: {str(e)}")
            self.cleanup_file(audio_filename)
            return new_history, "", None, f"⚠️ Video creation failed: {str(e)}"
        
        progress(0.9, desc="🎉 Finalizing...")
        
        # 4. 检查生成的文件是否存在
        if os.path.exists(video_filename):
            print(f"响应视频生成成功: {video_filename}")
            # 清理音频文件（保留视频文件）
            self.cleanup_file(audio_filename)
            return new_history, "", video_filename, f"✅ Successfully generated response video: {video_filename}"
        else:
            print("响应视频文件未找到")
            self.cleanup_file(audio_filename)
            return new_history, "", None, "⚠️ Response video file not found."
    
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
        return [], "", self.initial_video, "💬 Chat cleared! Ready for new conversation."

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
        # 🤖 Virtual Chat with AI Host
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
                    type="messages"
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
                - **TTS**: Test Mode (silent)
                - **Video**: Copy of base video
                - **GPU**: Auto-managed
                """)
        
        # 事件处理函数
        def handle_send(user_input, history):
            if not user_input.strip():
                return history, "", None, "⚠️ Please enter a message."
            
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
            3. **Response Videos** - New videos replace the welcome video
            4. **Video Controls** - Manual play/pause/stop buttons
            5. **Progress Tracking** - Shows processing steps
            
            ### How it Works:
            - The welcome video loads when you open the page
            - When you send a message, a new response video is generated
            - Use the video control buttons to play/pause/stop videos
            - Each conversation generates a unique response video
            
            ### Video Playback:
            - Click "Play Video" to start playback
            - Use "Pause" to pause the video
            - Use "Stop" to stop the video
            - Videos will auto-replace when new responses are generated
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
