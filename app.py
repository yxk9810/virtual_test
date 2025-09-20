import gradio as gr
import subprocess
import os
import time
import threading
from gemini_chat import GeminiChatSession
from system_prompt import prompt

class VirtualChatApp:
    def __init__(self):
        self.chat_session = GeminiChatSession(
            model="gemini-2.0-flash",
            system_instruction=prompt
        )
        self.chat_history = []
        
    def process_message(self, user_input, history):
        """处理用户消息的完整流程"""
        if not user_input.strip():
            return history, "", None
            
        # 1. 调用Gemini生成回复文本
        print(f"用户输入: {user_input}")
        ai_response = self.chat_session.send_message(user_input)
        
        if not ai_response:
            ai_response = "Sorry, I couldn't generate a response. Please try again."
        
        print(f"AI回复: {ai_response}")
        
        # 2. 生成音频文件
        audio_filename = f"audio_{int(time.time())}.wav"
        audio_cmd = f"python generate_audio.py \"{ai_response}\" {audio_filename}"
        
        print(f"执行音频生成命令: {audio_cmd}")
        try:
            result = subprocess.run(audio_cmd, shell=True, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"音频生成失败: {result.stderr}")
                return history + [[user_input, ai_response]], ai_response, None
        except subprocess.TimeoutExpired:
            print("音频生成超时")
            return history + [[user_input, ai_response]], ai_response, None
        
        # 3. 生成视频文件
        video_filename = f"video_{int(time.time())}.mp4"
        base_video_path = "/kaggle/input/audio-demo1/kling_20250806_Image_to_Video_this_woman_3742_0.mp4"
        video_cmd = f"python generate_video.py \"{base_video_path}\" {audio_filename} {video_filename}"
        
        print(f"执行视频生成命令: {video_cmd}")
        try:
            result = subprocess.run(video_cmd, shell=True, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"视频生成失败: {result.stderr}")
                return history + [[user_input, ai_response]], ai_response, None
        except subprocess.TimeoutExpired:
            print("视频生成超时")
            return history + [[user_input, ai_response]], ai_response, None
        
        # 4. 检查生成的文件是否存在
        if os.path.exists(video_filename):
            print(f"视频生成成功: {video_filename}")
            return history + [[user_input, ai_response]], ai_response, video_filename
        else:
            print("视频文件未找到")
            return history + [[user_input, ai_response]], ai_response, None
    
    def clear_chat(self):
        """清空聊天历史"""
        self.chat_session.clear_history()
        return [], "", None

def create_interface():
    app = VirtualChatApp()
    
    with gr.Blocks(title="Virtual Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Virtual Chat with AI")
        gr.Markdown("Chat with an AI virtual host! The AI will respond with both text and video.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # 聊天历史显示
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=400,
                    show_label=True
                )
                
                # 用户输入框
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Your Message",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                # 控制按钮
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            with gr.Column(scale=1):
                # 视频显示区域
                video_output = gr.Video(
                    label="AI Response Video",
                    height=400,
                    autoplay=True
                )
                
                # 状态显示
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready to chat!",
                    interactive=False,
                    lines=3
                )
        
        # 事件处理
        def handle_send(user_input, history):
            if not user_input.strip():
                return history, "", None, "Please enter a message."
            
            new_history, ai_response, video_file = app.process_message(user_input, history)
            
            if video_file:
                status_msg = f"✅ Generated video: {video_file}"
            else:
                status_msg = "⚠️ Video generation failed, but text response is ready."
            
            return new_history, "", video_file, status_msg
        
        def handle_clear():
            return app.clear_chat()
        
        # 绑定事件
        send_btn.click(
            handle_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, video_output, status_text]
        )
        
        msg_input.submit(
            handle_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, video_output, status_text]
        )
        
        clear_btn.click(
            handle_clear,
            outputs=[chatbot, msg_input, video_output]
        )
        
        # 示例
        gr.Markdown("""
        ### 💡 Tips:
        - Type your message and press Enter or click Send
        - The AI will respond in English with both text and video
        - Videos are generated using your specified base video and AI-generated audio
        - Use "Clear Chat" to reset the conversation
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
