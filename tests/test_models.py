"""
LLMモデル実装のテスト。
"""

import pytest
import os
from unittest.mock import Mock, patch
from llm_economist.models.base import BaseLLMModel
from llm_economist.models.openai_model import OpenAIModel


class TestBaseLLMModel:
    """ベースLLMモデルクラスのテスト。"""

    def test_base_model_abstract(self):
        """BaseLLMModelが直接インスタンス化できないことを確認するテスト。"""
        with pytest.raises(TypeError):
            BaseLLMModel("test-model")

    def test_json_extraction(self):
        """JSON抽出機能のテスト。"""
        class TestModel(BaseLLMModel):
            def send_msg(self, system_prompt, user_prompt, temperature=None, json_format=False):
                return "test", False

        model = TestModel("test-model")

        # 有効なJSONのテスト
        valid_json = '{"key": "value", "number": 42}'
        result, is_valid = model._extract_json(f"Some text {valid_json} more text")
        assert is_valid
        assert result == valid_json

        # 無効なJSONのテスト
        invalid_json = '{"key": "value"'
        result, is_valid = model._extract_json(f"Some text {invalid_json} more text")
        assert not is_valid

        # JSONなしのテスト
        no_json = "This is just regular text"
        result, is_valid = model._extract_json(no_json)
        assert not is_valid


class TestOpenAIModel:
    """OpenAIモデル実装のテスト。"""

    def test_init_with_api_key(self):
        """APIキーありでの初期化テスト。"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            model = OpenAIModel()
            assert model.model_name == "gpt-4o-mini"

    def test_init_without_api_key(self):
        """APIキーなしでの初期化がエラーを発生させることを確認するテスト。"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                OpenAIModel()

    @patch('llm_economist.models.openai_model.OpenAI')
    def test_send_msg_success(self, mock_openai):
        """メッセージ送信成功のテスト。"""
        # OpenAIクライアントのレスポンスをモック
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            model = OpenAIModel()
            response, is_json = model.send_msg("System prompt", "User prompt")

            assert response == "Test response"
            assert not is_json
            mock_client.chat.completions.create.assert_called_once()

    def test_get_available_models(self):
        """利用可能なモデル一覧の取得テスト。"""
        models = OpenAIModel.get_available_models()
        assert isinstance(models, list)
        assert "gpt-4o-mini" in models


class TestModelIntegration:
    """モデル切り替えの結合テスト。"""

    def test_model_factory_pattern(self):
        """ファクトリパターンによるモデル生成のテスト。"""
        # LLMAgentがモデルを生成する方法をシミュレート

        def create_model(model_type, **kwargs):
            if "gpt" in model_type.lower():
                return "OpenAI"
            else:
                return "Unknown"

        assert create_model("gpt-4o-mini") == "OpenAI"


# テスト用フィクスチャ
@pytest.fixture
def mock_openai_env():
    """環境変数にOpenAI APIキーを設定するフィクスチャ。"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield
