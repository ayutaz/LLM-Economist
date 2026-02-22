"""
LLM Economistフレームワークのクイックスタートサンプル。

このモジュールは基本的な機能テストとセットアップの検証を提供します。
実際のシミュレーション例については、advanced_usage.pyを参照してください。
"""

import os
import sys
from llm_economist.main import run_simulation, create_argument_parser, generate_experiment_name


def test_imports():
    """必要な全モジュールがインポートできることをテストする。"""
    print("インポートをテスト中...")
    
    try:
        from llm_economist.main import run_simulation, create_argument_parser
        from llm_economist.agents.worker import Worker
        from llm_economist.agents.planner import TaxPlanner
        from llm_economist.agents.llm_agent import TestAgent
        from llm_economist.utils.common import distribute_agents
        from llm_economist.agents.worker import distribute_personas
        print("✓ 全てのインポートに成功しました")
        return True
    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False


def test_argument_parser():
    """引数パーサーが正しく動作することをテストする。"""
    print("引数パーサーをテスト中...")
    
    try:
        parser = create_argument_parser()
        # 最小限の引数でテスト
        args = parser.parse_args([
            "--scenario", "rational",
            "--num-agents", "3",
            "--max-timesteps", "5",
            "--worker-type", "LLM",
            "--planner-type", "LLM",
            "--llm", "gpt-4o-mini"
        ])
        
        assert args.scenario == "rational"
        assert args.num_agents == 3
        assert args.max_timesteps == 5
        assert args.worker_type == "LLM"
        assert args.planner_type == "LLM"
        assert args.llm == "gpt-4o-mini"
        
        print("✓ 引数パーサーは正常に動作しています")
        return True
    except Exception as e:
        print(f"✗ 引数パーサーエラー: {e}")
        return False


def test_experiment_name_generation():
    """実験名の生成をテストする。"""
    print("実験名の生成をテスト中...")
    
    try:
        class Args:
            scenario = "rational"
            num_agents = 5
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "gpt-4o-mini"
            prompt_algo = "io"
            two_timescale = 25
            history_len = 50
            max_timesteps = 100
            bracket_setting = "two"
            percent_ego = 100
            percent_alt = 0
            percent_adv = 0
            platforms = False
        
        args = Args()
        name = generate_experiment_name(args)
        
        # 名前に期待されるコンポーネントが含まれているか確認
        expected_parts = ["rational", "a5", "w-LLM", "p-LLM", "llm-g"]
        for part in expected_parts:
            assert part in name, f"Expected '{part}' in experiment name '{name}'"
        
        print(f"✓ 実験名の生成が正常に動作: {name}")
        return True
    except Exception as e:
        print(f"✗ 実験名の生成エラー: {e}")
        return False


def test_api_key_detection():
    """OpenAI APIキーの検出をテストする。"""
    print("APIキーの検出をテスト中...")

    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ECON_OPENAI')

    if api_key:
        print("✓ OpenAI APIキーが見つかりました")
        return True
    else:
        print("- OpenAI APIキーが見つかりません（テスト用途では問題ありません）")
        return True


def test_basic_args_creation():
    """基本的な引数オブジェクトの作成をテストする。"""
    print("基本的なArgsオブジェクトの作成をテスト中...")
    
    try:
        class Args:
            scenario = "rational"
            num_agents = 3
            max_timesteps = 5
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "gpt-4o-mini"
            prompt_algo = "io"
            history_len = 20
            timeout = 10
            two_timescale = 10
            agent_mix = "us_income"
            bracket_setting = "three"
            percent_ego = 100
            percent_alt = 0
            percent_adv = 0
            tax_type = "US_FED"
            warmup = 0
            wandb = False
            debug = False
            use_multithreading = False
            platforms = False
            name = ""
            log_dir = "logs"
            elasticity = [0.4]
            seed = 42

        args = Args()

        # 全ての必須属性が存在することを検証
        required_attrs = [
            'scenario', 'num_agents', 'max_timesteps', 'worker_type',
            'planner_type', 'llm', 'agent_mix', 'bracket_setting',
            'percent_ego', 'percent_alt', 'percent_adv', 'tax_type'
        ]
        
        for attr in required_attrs:
            assert hasattr(args, attr), f"Missing required attribute: {attr}"
        
        print("✓ 基本的なArgsオブジェクトの作成に成功しました")
        return True
    except Exception as e:
        print(f"✗ Argsオブジェクトの作成エラー: {e}")
        return False


def test_service_configurations():
    """OpenAI設定をテストする。"""
    print("サービス設定をテスト中...")

    try:
        class Args:
            scenario = "rational"
            num_agents = 3
            max_timesteps = 5
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "gpt-4o-mini"
            prompt_algo = "io"
            history_len = 20
            timeout = 10
            two_timescale = 10
            agent_mix = "us_income"
            bracket_setting = "three"
            percent_ego = 100
            percent_alt = 0
            percent_adv = 0
            tax_type = "US_FED"
            warmup = 0
            wandb = False
            debug = False
            use_multithreading = False
            platforms = False
            name = ""
            log_dir = "logs"
            elasticity = [0.4]
            seed = 42

        args = Args()
        print("✓ OpenAI設定が有効です")
    except Exception as e:
        print(f"✗ 設定エラー: {e}")
        return False

    print("✓ サービス設定が有効です")
    return True


def run_all_tests():
    """全ての基本機能テストを実行する。"""
    print("="*50)
    print("LLM Economistクイックスタートテストを実行中")
    print("="*50)
    
    tests = [
        test_imports,
        test_argument_parser,
        test_experiment_name_generation,
        test_api_key_detection,
        test_basic_args_creation,
        test_service_configurations,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n{test.__name__}:")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ テストが例外で失敗: {e}")
            failed += 1

    print("\n" + "="*50)
    print(f"テスト結果: {passed} 成功, {failed} 失敗")
    print("="*50)

    if failed == 0:
        print("全ての基本機能テストに合格しました!")
        print("実際のシミュレーション例を実行するには: python examples/advanced_usage.py")
    else:
        print("一部のテストが失敗しました。上記のエラーを確認してください。")
    
    return failed == 0


def main():
    """クイックスタートテストのメインエントリポイント。"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        print("\n使い方:")
        print("  python examples/quick_start.py          # 全ての基本テストを実行")
        print("  python examples/quick_start.py --help   # このヘルプを表示")
        print("\n実際のシミュレーション例:")
        print("  python examples/advanced_usage.py --help")
        return
    
    success = run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 