"""
LLM EconomistフレームワークのOpenRouterモデル実装。
"""

from typing import Tuple, Optional
import os
import requests
import json
from time import sleep
from .base import BaseLLMModel


class OpenRouterModel(BaseLLMModel):
    """OpenRouter APIを通じて複数のモデルにアクセスするOpenRouterモデル実装。"""

    def __init__(self, model_name: str = "meta-llama/llama-3.1-8b-instruct",
                 api_key: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.7):
        """
        OpenRouterモデルを初期化する。

        Args:
            model_name: OpenRouterで使用するモデル名
            api_key: OpenRouter APIキー (Noneの場合、OPENROUTER_API_KEY環境変数を参照)
            max_tokens: 生成する最大トークン数
            temperature: サンプリングの温度パラメータ
        """
        super().__init__(model_name, max_tokens, temperature)
        
        # パラメータまたは環境変数からAPIキーを取得
        if api_key is None:
            api_key = os.getenv('OPENROUTER_API_KEY')

        if not api_key:
            raise ValueError("OpenRouter APIキーが見つかりません。OPENROUTER_API_KEY環境変数を設定するか、api_keyパラメータを渡してください。")
        
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        
        # OpenRouter APIのヘッダー
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/sethkarten/LLMEconomist",
            "X-Title": "LLM Economist",
            "Content-Type": "application/json"
        }
        
    def send_msg(self, system_prompt: str, user_prompt: str,
                 temperature: Optional[float] = None,
                 json_format: bool = False) -> Tuple[str, bool]:
        """
        OpenRouter APIにメッセージを送信してレスポンスを取得する。

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
                # リクエストペイロードの準備
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": self.max_tokens
                }
                
                # 要求された場合JSONフォーマットを追加 (互換性のあるモデル用)
                if json_format:
                    payload["response_format"] = {"type": "json_object"}
                
                # API呼び出しの実行
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                response.raise_for_status()
                result = response.json()
                
                if 'choices' not in result or len(result['choices']) == 0:
                    raise Exception(f"レスポンスの選択肢が返されませんでした: {result}")
                
                message = result['choices'][0]['message']['content']
                
                if not self._validate_response(message):
                    self.logger.warning(f"無効なレスポンスを受信: {message}")
                    retry_count += 1
                    continue

                # 要求された場合JSONを抽出
                if json_format:
                    return self._extract_json(message)
                
                return message, False
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # レート制限
                    self.logger.warning(f"レート制限に到達: {e}")
                    self._handle_rate_limit(retry_count, max_retries)
                    retry_count += 1
                else:
                    self.logger.error(f"OpenRouter API HTTPエラー: {e}")
                    raise

            except requests.exceptions.RequestException as e:
                self.logger.error(f"OpenRouter APIリクエストエラー: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                sleep(1)
                
            except Exception as e:
                self.logger.error(f"OpenRouter API呼び出しエラー: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                sleep(1)
        
        raise Exception(f"{max_retries}回のリトライ後にレスポンスの取得に失敗しました")

    def get_models(self) -> list:
        """OpenRouterから利用可能なモデルのリストを取得する。"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['data']
        except Exception as e:
            self.logger.error(f"モデル一覧取得エラー: {e}")
            return []
    
    @classmethod
    def get_popular_models(cls):
        """OpenRouterで利用可能な代表的なモデルのリストを取得する。"""
        return [
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-405b-instruct",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-haiku",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4-turbo",
            "google/gemini-pro-1.5",
            "google/gemini-flash-1.5",
            "mistralai/mistral-7b-instruct",
            "mistralai/mixtral-8x7b-instruct",
            "cohere/command-r-plus",
            "perplexity/llama-3.1-sonar-large-128k-online"
        ]
    
    def check_model_availability(self, model_name: str) -> bool:
        """特定のモデルがOpenRouterで利用可能かどうかを確認する。"""
        try:
            models = self.get_models()
            return any(model['id'] == model_name for model in models)
        except:
            return False 