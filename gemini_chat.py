from google.genai import Client, types
from google.genai.types import UserContent, ModelContent, Part
from system_prompt import *

# 初始化客户端
client = Client(api_key="AIzaSyDYJXsBaGD4KQ2L-yOibBZwcCCgEjfMexg")

class GeminiChatSession:
    def __init__(self, model="gemini-2.0-flash", system_instruction=None):
        self.client = client
        self.model = model
        self.system_instruction = system_instruction
        self.history = []

    def add_user_message(self, message):
        """添加用户消息到历史记录"""
        self.history.append(UserContent(parts=[Part(text=message)]))

    def add_model_message(self, message):
        """添加模型回复到历史记录"""
        self.history.append(ModelContent(parts=[Part(text=message)]))

    def send_message(self, message, temperature=0.7, max_output_tokens=1024):
        """发送消息并获取回复"""
        # 将用户消息添加到历史
        self.add_user_message(message)

        # 创建聊天会话配置
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        # 如果有系统指令，添加到配置中
        if self.system_instruction:
            config.system_instruction = self.system_instruction

        try:
            # 创建聊天会话
            chat_session = self.client.chats.create(
                model=self.model,
                history=self.history.copy(),  # 传递当前历史记录的副本
                config=config
            )

            # 发送消息并获取回复
            response = chat_session.send_message(message)

            # 将模型回复添加到历史
            self.add_model_message(response.text)
            print(response.usage_metadata)
            return response.text

        except Exception as e:
            print(f"发送消息时出错: {e}")
            return None

    def get_history(self):
        """获取聊天历史"""
        return self.history

    def clear_history(self):
        """清空聊天历史"""
        self.history = []
