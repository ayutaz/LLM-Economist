"""
LLM EconomistフレームワークのvLLMモデル実装。
"""

from typing import Tuple, Optional
import os
import requests
import json
from openai import OpenAI, RateLimitError
from time import sleep
from .base import BaseLLMModel


class VLLMModel(BaseLLMModel):
    """ローカルおよびリモートvLLMデプロイメント用のvLLMモデル実装。"""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 base_url: str = "http://localhost:8000",
                 api_key: str = "economist",
                 max_tokens: int = 1000, temperature: float = 0.7):
        """
        vLLMモデルを初期化する。

        Args:
            model_name: 使用するモデル名
            base_url: vLLMサーバーのベースURL
            api_key: 認証用APIキー
            max_tokens: 生成する最大トークン数
            temperature: サンプリングの温度パラメータ
        """
        super().__init__(model_name, max_tokens, temperature)
        
        self.base_url = base_url
        self.api_key = api_key
        
        # vLLM互換のためOpenAIクライアントを初期化
        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{base_url}/v1"
        )
        
        # 後方互換性のためのモデル名マッピング
        self.model_mapping = {
            'llama3:8b': 'meta-llama/Llama-3.1-8B-Instruct',
            'llama3:70b': 'meta-llama/Llama-3.1-70B-Instruct',
            'gemma3:27b': 'google/gemma-3-27b-it',
        }
        
        # マッピングされたモデル名が利用可能な場合使用
        if model_name in self.model_mapping:
            self.model_name = self.model_mapping[model_name]
        
    def send_msg(self, system_prompt: str, user_prompt: str,
                 temperature: Optional[float] = None,
                 json_format: bool = False) -> Tuple[str, bool]:
        """
        vLLMサーバーにメッセージを送信してレスポンスを取得する。

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
                # vLLMでは結合プロンプトでcompletionsエンドポイントを使用
                combined_prompt = f"{system_prompt}\n{user_prompt}"
                
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=combined_prompt,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    stop=self.stop_tokens,
                    stream=False
                )
                
                message = response.choices[0].text
                
                if not self._validate_response(message):
                    self.logger.warning(f"無効なレスポンスを受信: {message}")
                    retry_count += 1
                    continue

                # 要求された場合JSONを抽出
                if json_format:
                    return self._extract_json(message)

                return message, False

            except RateLimitError as e:
                self.logger.warning(f"レート制限に到達: {e}")
                self._handle_rate_limit(retry_count, max_retries)
                retry_count += 1

            except Exception as e:
                self.logger.error(f"vLLM API呼び出しエラー: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                sleep(1)

        raise Exception(f"{max_retries}回のリトライ後にレスポンスの取得に失敗しました")

    def check_health(self) -> bool:
        """vLLMサーバーの正常性を確認する。"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    @classmethod
    def get_available_models(cls):
        """vLLMで動作する代表的なモデルのリストを取得する。"""
        return [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct", 
            "meta-llama/Llama-3.1-405B-Instruct",
            "google/gemma-3-27b-it",
            "microsoft/DialoGPT-large",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ]


class OllamaModel(BaseLLMModel):
    """ローカルOllamaデプロイメント用のOllamaモデル実装。"""

    def __init__(self, model_name: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434",
                 max_tokens: int = 1000, temperature: float = 0.7):
        """
        Ollamaモデルを初期化する。

        Args:
            model_name: 使用するOllamaモデル名
            base_url: OllamaサーバーのベースURL
            max_tokens: 生成する最大トークン数
            temperature: サンプリングの温度パラメータ
        """
        super().__init__(model_name, max_tokens, temperature)
        
        self.base_url = base_url
        
        # 依存関係の問題を避けるためここでollamaをインポート
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
        except ImportError:
            raise ImportError("ollamaをインストールしてください: pip install ollama")
        
    def send_msg(self, system_prompt: str, user_prompt: str,
                 temperature: Optional[float] = None,
                 json_format: bool = False) -> Tuple[str, bool]:
        """
        Ollamaサーバーにメッセージを送信してレスポンスを取得する。

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
                response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": temperature,
                        "num_predict": self.max_tokens
                    }
                )
                
                message = response['message']['content']
                
                if not self._validate_response(message):
                    self.logger.warning(f"無効なレスポンスを受信: {message}")
                    retry_count += 1
                    continue

                # 要求された場合JSONを抽出
                if json_format:
                    return self._extract_json(message)

                return message, False

            except Exception as e:
                self.logger.error(f"Ollama API呼び出しエラー: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                sleep(1)

        raise Exception(f"{max_retries}回のリトライ後にレスポンスの取得に失敗しました")

    @classmethod
    def get_available_models(cls):
        """Ollamaで動作する代表的なモデルのリストを取得する。"""
        return [
            "llama3.1:8b",
            "llama3.1:70b", 
            "llama3.1:405b",
            "gemma3:27b",
            "mistral:7b",
            "codellama:7b"
        ] 