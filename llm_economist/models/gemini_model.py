"""
LLM EconomistフレームワークのGeminiモデル実装。
"""

from typing import Tuple, Optional
import os
import json
from time import sleep
from .base import BaseLLMModel


class GeminiModel(BaseLLMModel):
    """GoogleのGemini APIを使用したGeminiモデル実装。"""

    def __init__(self, model_name: str = "gemini-1.5-flash",
                 api_key: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.7):
        """
        Geminiモデルを初期化する。

        Args:
            model_name: 使用するGeminiモデル名
            api_key: Google APIキー (Noneの場合、GOOGLE_API_KEY環境変数を参照)
            max_tokens: 生成する最大トークン数
            temperature: サンプリングの温度パラメータ
        """
        super().__init__(model_name, max_tokens, temperature)
        
        # パラメータまたは環境変数からAPIキーを取得
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

        if not api_key:
            raise ValueError("Google APIキーが見つかりません。GOOGLE_API_KEY環境変数を設定するか、api_keyパラメータを渡してください。")
        
        self.api_key = api_key
        
        # Google AI SDKのインポート
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai
        except ImportError:
            raise ImportError("Google AI SDKをインストールしてください: pip install google-generativeai")

        # モデルの初期化
        self.model = self.client.GenerativeModel(model_name)
        
    def send_msg(self, system_prompt: str, user_prompt: str,
                 temperature: Optional[float] = None,
                 json_format: bool = False) -> Tuple[str, bool]:
        """
        Gemini APIにメッセージを送信してレスポンスを取得する。

        Args:
            system_prompt: コンテキストを設定するシステムプロンプト
            user_prompt: ユーザーのプロンプト/質問
            temperature: この呼び出しの温度オーバーライド
            json_format: JSONフォーマットのレスポンスを要求するかどうか

        Returns:
            (レスポンステキスト, JSONが有効か) のタプル
        """
        if temperature is None:
            temperature = self.temperature

        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                # システムプロンプトとユーザープロンプトを結合
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"

                # 要求された場合JSONフォーマット指示を追加
                if json_format:
                    combined_prompt += "\n\nPlease respond in valid JSON format."
                
                # 生成パラメータの設定
                generation_config = self.client.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=self.max_tokens,
                    candidate_count=1
                )
                
                # レスポンスの生成
                response = self.model.generate_content(
                    combined_prompt,
                    generation_config=generation_config
                )
                
                message = response.text
                
                if not self._validate_response(message):
                    self.logger.warning(f"無効なレスポンスを受信: {message}")
                    retry_count += 1
                    continue

                # 要求された場合JSONを抽出
                if json_format:
                    return self._extract_json(message)
                
                return message, False
                
            except Exception as e:
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    self.logger.warning(f"レート制限またはクォータ超過: {e}")
                    self._handle_rate_limit(retry_count, max_retries)
                    retry_count += 1
                else:
                    self.logger.error(f"Gemini API呼び出しエラー: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                    sleep(1)
        
        raise Exception(f"{max_retries}回のリトライ後にレスポンスの取得に失敗しました")

    @classmethod
    def get_available_models(cls):
        """利用可能なGeminiモデルのリストを取得する。"""
        return [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
            "gemini-1.0-pro-latest"
        ]
    
    def list_models(self):
        """利用可能な全モデルを動的にリストする。"""
        try:
            models = list(self.client.list_models())
            return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
        except Exception as e:
            self.logger.error(f"モデル一覧取得エラー: {e}")
            return self.get_available_models()


class GeminiModelViaOpenRouter(BaseLLMModel):
    """OpenRouterをプロキシとして使用するGeminiモデル実装。"""

    def __init__(self, model_name: str = "google/gemini-flash-1.5",
                 api_key: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.7):
        """
        OpenRouter経由でGeminiモデルを初期化する。

        Args:
            model_name: OpenRouter上のGeminiモデル名
            api_key: OpenRouter APIキー (Noneの場合、OPENROUTER_API_KEY環境変数を参照)
            max_tokens: 生成する最大トークン数
            temperature: サンプリングの温度パラメータ
        """
        super().__init__(model_name, max_tokens, temperature)
        
        # OpenRouterモデルのインポート
        from .openrouter_model import OpenRouterModel
        
        self.openrouter_client = OpenRouterModel(
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
    def send_msg(self, system_prompt: str, user_prompt: str,
                 temperature: Optional[float] = None,
                 json_format: bool = False) -> Tuple[str, bool]:
        """
        OpenRouter経由でGemini APIにメッセージを送信してレスポンスを取得する。

        Args:
            system_prompt: コンテキストを設定するシステムプロンプト
            user_prompt: ユーザーのプロンプト/質問
            temperature: この呼び出しの温度オーバーライド
            json_format: JSONフォーマットのレスポンスを要求するかどうか

        Returns:
            (レスポンステキスト, JSONが有効か) のタプル
        """
        return self.openrouter_client.send_msg(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            json_format=json_format
        )
    
    @classmethod
    def get_available_models(cls):
        """OpenRouter上で利用可能なGeminiモデルのリストを取得する。"""
        return [
            "google/gemini-pro-1.5",
            "google/gemini-flash-1.5",
            "google/gemini-pro"
        ] 