"""
LLMモデル実装のテスト。
"""

import pytest
import os
from unittest.mock import Mock, patch
from llm_economist.models.base import BaseLLMModel
from llm_economist.models.openai_model import OpenAIModel
from llm_economist.models.vllm_model import VLLMModel, OllamaModel
from llm_economist.models.openrouter_model import OpenRouterModel
from llm_economist.models.gemini_model import GeminiModel


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


class TestVLLMModel:
    """vLLMモデル実装のテスト。"""
    
    @patch('llm_economist.models.vllm_model.OpenAI')
    def test_init(self, mock_openai):
        """vLLMモデル初期化のテスト。"""
        model = VLLMModel()
        assert model.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert model.base_url == "http://localhost:8000"
    
    def test_model_mapping(self):
        """モデル名マッピングのテスト。"""
        with patch('llm_economist.models.vllm_model.OpenAI'):
            model = VLLMModel(model_name="llama3:8b")
            assert model.model_name == "meta-llama/Llama-3.1-8B-Instruct"
    
    @patch('llm_economist.models.vllm_model.requests.get')
    def test_health_check(self, mock_get):
        """ヘルスチェック機能のテスト。"""
        with patch('llm_economist.models.vllm_model.OpenAI'):
            model = VLLMModel()
            
            # 正常なサーバーのテスト
            mock_get.return_value.status_code = 200
            assert model.check_health()
            
            # 異常なサーバーのテスト
            mock_get.side_effect = Exception("Connection error")
            assert not model.check_health()


class TestOllamaModel:
    """Ollamaモデル実装のテスト。"""

    def test_init_missing_ollama(self):
        """ollamaパッケージが未インストールの場合の初期化テスト。"""
        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(ImportError, match="Please install ollama"):
                OllamaModel()


class TestOpenRouterModel:
    """OpenRouterモデル実装のテスト。"""

    def test_init_with_api_key(self):
        """APIキーありでの初期化テスト。"""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            model = OpenRouterModel()
            assert model.api_key == "test-key"

    def test_init_without_api_key(self):
        """APIキーなしでの初期化がエラーを発生させることを確認するテスト。"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key not found"):
                OpenRouterModel()
    
    def test_get_popular_models(self):
        """人気モデル一覧の取得テスト。"""
        models = OpenRouterModel.get_popular_models()
        assert isinstance(models, list)
        assert "meta-llama/llama-3.1-8b-instruct" in models
    
    @patch('llm_economist.models.openrouter_model.requests.post')
    def test_send_msg_success(self, mock_post):
        """メッセージ送信成功のテスト。"""
        # 成功レスポンスをモック
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            model = OpenRouterModel()
            response, is_json = model.send_msg("System prompt", "User prompt")
            
            assert response == "Test response"
            assert not is_json


class TestGeminiModel:
    """Geminiモデル実装のテスト。"""

    def test_init_without_api_key(self):
        """APIキーなしでの初期化がエラーを発生させることを確認するテスト。"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google API key not found"):
                GeminiModel()
    
    def test_init_missing_google_ai(self):
        """google.generativeaiパッケージが未インストールの場合の初期化テスト。"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch('builtins.__import__', side_effect=ImportError):
                with pytest.raises(ImportError, match="Please install Google AI SDK"):
                    GeminiModel()
    
    def test_get_available_models(self):
        """利用可能なモデル一覧の取得テスト。"""
        models = GeminiModel.get_available_models()
        assert isinstance(models, list)
        assert "gemini-1.5-flash" in models


class TestModelIntegration:
    """モデル切り替えの結合テスト。"""

    def test_model_factory_pattern(self):
        """ファクトリパターンによるモデル生成のテスト。"""
        # LLMAgentがモデルを生成する方法をシミュレート
        
        def create_model(model_type, **kwargs):
            if "gpt" in model_type.lower():
                return "OpenAI"
            elif "llama" in model_type.lower():
                return "vLLM"
            elif "claude" in model_type.lower():
                return "OpenRouter"
            elif "gemini" in model_type.lower():
                return "Gemini"
            else:
                return "Unknown"
        
        assert create_model("gpt-4o-mini") == "OpenAI"
        assert create_model("llama3:8b") == "vLLM"
        assert create_model("claude-3.5-sonnet") == "OpenRouter"
        assert create_model("gemini-1.5-flash") == "Gemini"


# テスト用フィクスチャ
@pytest.fixture
def mock_openai_env():
    """環境変数にOpenAI APIキーを設定するフィクスチャ。"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


@pytest.fixture
def mock_openrouter_env():
    """環境変数にOpenRouter APIキーを設定するフィクスチャ。"""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        yield


@pytest.fixture
def mock_google_env():
    """環境変数にGoogle APIキーを設定するフィクスチャ。"""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
        yield 