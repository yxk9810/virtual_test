import gradio as gr
import os

def test_video():
    video_path = 'welcome_video.mp4'
    if os.path.exists(video_path):
        print(f"视频文件存在: {video_path}")
        return video_path
    else:
        print("视频文件不存在")
        return None

with gr.Blocks(title="视频播放测试") as demo:
    gr.Markdown("# 🎬 视频播放测试")
    
    with gr.Row():
        with gr.Column():
            video = gr.Video(
                label="测试视频",
                height=400,
                autoplay=True,
                show_share_button=False
            )
            
            btn = gr.Button("重新加载视频", variant="primary")
            btn.click(test_video, outputs=video)
            
            # 初始加载视频
            demo.load(test_video, outputs=video)

if __name__ == "__main__":
    print("启动视频播放测试...")
    demo.launch(server_port=7862, share=False)
