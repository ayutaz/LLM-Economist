"""
高度な使用シナリオの結合テスト。

実際のAPIコールを使用してシミュレーションを実行し、
LLM Economistの完全な機能を検証するテスト。
"""

import pytest
import os
from unittest.mock import patch


class TestAdvancedUsageIntegration:
    """高度な使用シナリオの結合テスト。"""

    def test_advanced_usage_imports(self):
        """高度な使用例の関数がインポートできることを確認するテスト。"""
        from examples.advanced_usage import (
            test_rational_openai,
            test_bounded_rationality,
            test_democratic_scenario,
            test_fixed_workers,
            main
        )

        # 呼び出し可能であることを検証
        assert callable(test_rational_openai)
        assert callable(test_bounded_rationality)
        assert callable(test_democratic_scenario)
        assert callable(test_fixed_workers)
        assert callable(main)

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI APIキーが利用不可")
    def test_rational_openai_simulation(self):
        """OpenAIを使用した合理的シナリオのテスト（APIキーが必要）。"""
        from examples.advanced_usage import test_rational_openai

        # APIキーが利用可能であればエラーなしで実行されるはず
        try:
            test_rational_openai()
        except Exception as e:
            # APIレート制限や一時的な障害を許容
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                pytest.skip(f"API rate limit or quota exceeded: {e}")
            else:
                raise

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI APIキーが利用不可")
    def test_bounded_rationality_simulation(self):
        """限定合理性シナリオのテスト（APIキーが必要）。"""
        from examples.advanced_usage import test_bounded_rationality

        # APIキーが利用可能であればエラーなしで実行されるはず
        try:
            test_bounded_rationality()
        except Exception as e:
            # APIレート制限や一時的な障害を許容
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                pytest.skip(f"API rate limit or quota exceeded: {e}")
            else:
                raise

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI APIキーが利用不可")
    def test_democratic_scenario_simulation(self):
        """民主的シナリオのテスト（APIキーが必要）。"""
        from examples.advanced_usage import test_democratic_scenario

        # APIキーが利用可能であればエラーなしで実行されるはず
        try:
            test_democratic_scenario()
        except Exception as e:
            # APIレート制限や一時的な障害を許容
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                pytest.skip(f"API rate limit or quota exceeded: {e}")
            else:
                raise

    def test_fixed_workers_simulation(self):
        """固定ワーカーシナリオのテスト（APIキー不要）。"""
        from examples.advanced_usage import test_fixed_workers

        # 固定エージェントを使用するため常に動作するはず
        test_fixed_workers()


class TestAdvancedUsageCommandLine:
    """高度な使用例のコマンドラインインターフェースのテスト。"""

    def test_help_command(self):
        """ヘルプコマンドが動作することを確認するテスト。"""
        import subprocess
        import sys

        result = subprocess.run([
            sys.executable, "examples/advanced_usage.py", "--help"
        ], capture_output=True, text=True)

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "scenarios" in result.stdout.lower() or "commands" in result.stdout.lower()

    def test_invalid_scenario(self):
        """無効なシナリオが適切に処理されることを確認するテスト。"""
        import subprocess
        import sys

        result = subprocess.run([
            sys.executable, "examples/advanced_usage.py", "invalid_scenario"
        ], capture_output=True, text=True)

        assert result.returncode == 1
        assert "invalid scenario" in result.stderr.lower() or "unknown command" in result.stdout.lower()

    def test_list_scenarios(self):
        """全シナリオがヘルプに表示されることを確認するテスト。"""
        import subprocess
        import sys

        result = subprocess.run([
            sys.executable, "examples/advanced_usage.py", "--help"
        ], capture_output=True, text=True)

        expected_scenarios = [
            "rational", "bounded", "democratic", "fixed"
        ]

        for scenario in expected_scenarios:
            assert scenario in result.stdout.lower()


class TestAdvancedUsageConfiguration:
    """高度な使用シナリオの設定とセットアップのテスト。"""

    def test_all_scenarios_have_20_timesteps(self):
        """全シナリオが20タイムステップで設定されていることを確認するテスト。"""
        from examples.advanced_usage import (
            test_rational_openai,
            test_bounded_rationality,
            test_democratic_scenario,
            test_fixed_workers
        )

        # リフレクションを使用して各関数のArgsクラスを確認
        import inspect

        functions = [
            test_rational_openai,
            test_bounded_rationality,
            test_democratic_scenario,
            test_fixed_workers
        ]

        for func in functions:
            source = inspect.getsource(func)
            # ソースコードにmax_timesteps = 20が含まれていることを確認
            assert "max_timesteps = 20" in source, f"Function {func.__name__} doesn't have max_timesteps = 20"

    def test_scenario_diversity(self):
        """異なるシナリオが異なる設定を持つことを確認するテスト。"""
        from examples.advanced_usage import (
            test_rational_openai,
            test_bounded_rationality,
            test_democratic_scenario,
            test_fixed_workers
        )

        # 異なるシナリオが異なる設定を持つことを確認
        # 基本的な健全性チェック
        import inspect

        rational_source = inspect.getsource(test_rational_openai)
        bounded_source = inspect.getsource(test_bounded_rationality)
        democratic_source = inspect.getsource(test_democratic_scenario)
        fixed_source = inspect.getsource(test_fixed_workers)

        # Rationalはscenario = "rational"を持つはず
        assert 'scenario = "rational"' in rational_source

        # Boundedはscenario = "bounded"を持つはず
        assert 'scenario = "bounded"' in bounded_source

        # Democraticはscenario = "democratic"を持つはず
        assert 'scenario = "democratic"' in democratic_source

        # Fixedはworker_type = "FIXED"を持つはず
        assert 'worker_type = "FIXED"' in fixed_source
        assert 'planner_type = "FIXED"' in fixed_source
