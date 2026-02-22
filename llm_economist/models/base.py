"""
LLM EconomistフレームワークのLLMモデル基底クラス。
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import logging
import time
from time import sleep


class BaseLLMModel(ABC):
    """全LLMモデル実装の基底クラス。"""

    def __init__(self, model_name: str, max_tokens: int = 1000, temperature: float = 0.7):
        """
        LLMモデルの基底クラスを初期化する。

        Args:
            model_name: 使用するモデル名
            max_tokens: 生成する最大トークン数
            temperature: サンプリングの温度パラメータ (0.0 ~ 1.0)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        self.stop_tokens = ['}']
        
    @abstractmethod
    def send_msg(self, system_prompt: str, user_prompt: str,
                 temperature: Optional[float] = None,
                 json_format: bool = False) -> Tuple[str, bool]:
        """
        LLMにメッセージを送信してレスポンスを取得する。

        Args:
            system_prompt: コンテキストを設定するシステムプロンプト
            user_prompt: ユーザーのプロンプト/質問
            temperature: この呼び出しの温度オーバーライド
            json_format: JSONフォーマットのレスポンスを要求するかどうか

        Returns:
            (レスポンステキスト, JSONが有効か) のタプル
        """
        pass
        
    def _handle_rate_limit(self, retry_count: int = 0, max_retries: int = 3):
        """レート制限を指数バックオフで処理する。"""
        if retry_count >= max_retries:
            raise Exception(f"最大リトライ回数 ({max_retries}) に達しました")
            
        wait_time = 2 ** retry_count
        self.logger.warning(f"レート制限中、{wait_time}秒待機します...")
        time.sleep(wait_time)
        
    def _extract_json(self, message: str) -> Tuple[str, bool]:
        """メッセージ文字列からJSONを抽出する。"""
        try:
            json_start = message.find('{')
            json_end = message.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return message, False
                
            json_str = message[json_start:json_end]
            if len(json_str) > 0:
                # 基本的なバリデーション - パースを試行
                import json
                json.loads(json_str)  # 無効な場合は例外を投げる
                return json_str, True
        except (ValueError, json.JSONDecodeError):
            pass
            
        return message, False
        
    def _validate_response(self, response: str) -> bool:
        """レスポンスが妥当かどうかを検証する。"""
        if not response or len(response.strip()) == 0:
            return False
        return True 