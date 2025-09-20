#!/usr/bin/env python3
"""
测试Gemini Chat功能
"""

import os
import sys

def test_gemini_chat():
    """测试Gemini Chat基本功能"""
    print("🧪 测试Gemini Chat功能...")
    
    try:
        # 导入Gemini Chat模块
        from gemini_chat import GeminiChatSession
        from system_prompt import prompt
        
        print("✅ 成功导入Gemini Chat模块")
        
        # 创建聊天会话
        chat_session = GeminiChatSession(
            model="gemini-2.0-flash",
            system_instruction=prompt
        )
        
        print("✅ 成功创建聊天会话")
        
        # 测试发送消息
        test_message = "Hello, how are you today?"
        print(f"📤 发送测试消息: {test_message}")
        
        response = chat_session.send_message(test_message)
        
        if response:
            print(f"✅ 收到回复: {response}")
            print("✅ Gemini Chat功能正常！")
            return True
        else:
            print("❌ 未收到回复")
            return False
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保安装了必要的依赖包:")
        print("pip install google-genai")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_system_prompt():
    """测试系统提示"""
    print("🧪 测试系统提示...")
    
    try:
        from system_prompt import prompt
        
        if prompt and len(prompt) > 0:
            print("✅ 系统提示加载成功")
            print(f"📝 提示内容: {prompt[:100]}...")
            return True
        else:
            print("❌ 系统提示为空")
            return False
            
    except Exception as e:
        print(f"❌ 系统提示测试失败: {e}")
        return False

def test_dependencies():
    """测试依赖包"""
    print("🧪 测试依赖包...")
    
    required_packages = [
        "google.genai",
        "gradio"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 需要安装的包: {', '.join(missing_packages)}")
        print("安装命令:")
        print("pip install google-genai gradio")
        return False
    else:
        print("✅ 所有依赖包都已安装")
        return True

if __name__ == "__main__":
    print("🚀 开始测试Gemini Chat功能...")
    print("=" * 50)
    
    # 测试依赖包
    deps_ok = test_dependencies()
    print()
    
    if not deps_ok:
        print("❌ 依赖包测试失败，请先安装必要的包")
        sys.exit(1)
    
    # 测试系统提示
    prompt_ok = test_system_prompt()
    print()
    
    # 测试Gemini Chat
    chat_ok = test_gemini_chat()
    print()
    
    print("=" * 50)
    if deps_ok and prompt_ok and chat_ok:
        print("🎉 所有测试通过！Gemini Chat功能正常！")
        print("\n📖 使用方法:")
        print("1. 运行完整应用: python app.py")
        print("2. 运行增强版应用: python enhanced_app.py")
        print("3. 直接测试聊天: python test_gemini_chat.py")
    else:
        print("❌ 部分测试失败，请检查配置")
        sys.exit(1)
