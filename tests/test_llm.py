"""models.llm 客户端配置的轻量测试（不触发模型加载）。"""

from financial_risk_ai_platform.models.llm import get_client


def test_get_client_is_local_and_bypasses_proxy():
    client = get_client()
    assert client._client.base_url.host in {"127.0.0.1", "localhost"}
    assert client._client.trust_env is False
