"""本地 Ollama LLM 客户端封装。

本项目默认连接本地 Ollama。注意：本机开启了 macOS 系统级代理
(127.0.0.1:1087 http / 1088 socks)，httpx 会读取系统代理配置，把
localhost 请求也塞进代理，导致连本地 Ollama 返回 503。因此这里显式
trust_env=False 直连本地 Ollama，绕过系统代理。
"""
from __future__ import annotations

import ollama

DEFAULT_HOST = "http://127.0.0.1:11434"


def get_client() -> ollama.Client:
    """返回直连本地 Ollama 的客户端（绕过系统代理）。"""
    return ollama.Client(host=DEFAULT_HOST, trust_env=False)


def chat(model: str, content: str, *, system: str | None = None) -> str:
    """发送一条消息，返回模型文本回复。"""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    resp = get_client().chat(model=model, messages=messages)
    return resp["message"]["content"]
