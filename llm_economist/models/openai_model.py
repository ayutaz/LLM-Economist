"""
LLM EconomistフレームワークのOpenAIモデル実装。
"""

from typing import Tuple, Optional
import os
from openai import OpenAI, RateLimitError
from time import sleep
from .base import BaseLLMModel


class OpenAIModel(BaseLLMModel):
    """OpenAI APIを使用したOpenAIモデル実装。"""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 max_tokens: int = 1000, temperature: float = 0.7):
        """
        OpenAIモデルを初期化する。

        Args:
            model_name: 使用するOpenAIモデル名
            api_key: OpenAI APIキー (Noneの場合、OPENAI_API_KEY環境変数を参照)
            max_tokens: 生成する最大トークン数
            temperature: サンプリングの温度パラメータ
        """
        super().__init__(model_name, max_tokens, temperature)
        
        # パラメータまたは環境変数からAPIキーを取得
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ECON_OPENAI')

        if not api_key:
            raise ValueError("OpenAI APIキーが見つかりません。OPENAI_API_KEY環境変数を設定するか、api_keyパラメータを渡してください。")
        
        self.client = OpenAI(api_key=api_key)
        
    def send_msg(self, system_prompt: str, user_prompt: str,
                 temperature: Optional[float] = None,
                 json_format: bool = False) -> Tuple[str, bool]:
        """
        OpenAI APIにメッセージを送信してレスポンスを取得する。

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
                # リクエストの準備
                request_params = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": self.max_tokens
                }
                
                # 要求された場合JSONフォーマットを追加
                if json_format:
                    request_params["response_format"] = {"type": "json_object"}
                
                # API呼び出しの実行
                response = self.client.chat.completions.create(**request_params)
                
                message = response.choices[0].message.content
                
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
                self.logger.error(f"OpenAI API呼び出しエラー: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                sleep(1)
        
        raise Exception(f"{max_retries}回のリトライ後にレスポンスの取得に失敗しました")

    @classmethod
    def get_available_models(cls):
        """利用可能なOpenAIモデルのリストを取得する。"""
        return [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ] 