"""
LLM Economistの高度な使用例。

このスクリプトは、異なるシナリオでの実際のシミュレーション実行を示します。
全てのシミュレーションはテスト目的で20タイムステップで実行されます。
"""

import os
import sys
from llm_economist.main import run_simulation


def test_rational_openai():
    """OpenAI GPT-4o-miniを使用した合理的シナリオをテストする。"""
    print("OpenAI GPT-4o-miniで合理的シナリオを実行中...")
    
    class Args:
        scenario = "rational"
        num_agents = 3
        max_timesteps = 20
        worker_type = "LLM"
        planner_type = "LLM"
        llm = "gpt-4o-mini"
        port = 8000
        service = "vllm"
        use_openrouter = False
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
    
    # OpenAI APIキーが設定されていることを確認
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ECON_OPENAI'):
        print("OPENAI_API_KEY環境変数を設定してください")
        print("例: export OPENAI_API_KEY=your_api_key_here")
        return False

    try:
        run_simulation(args)
        print("✓ 合理的シナリオのシミュレーションが正常に完了しました")
        return True
    except Exception as e:
        print(f"✗ 合理的シナリオのシミュレーションが失敗しました: {e}")
        return False


def test_bounded_rationality():
    """限定合理性シナリオをテストする。"""
    print("限定合理性シナリオを実行中...")
    
    class Args:
        scenario = "bounded"
        num_agents = 3
        max_timesteps = 20
        worker_type = "LLM"
        planner_type = "LLM"
        llm = "gpt-4o-mini"
        port = 8000
        service = "vllm"
        use_openrouter = False
        prompt_algo = "io"
        history_len = 20
        timeout = 10
        two_timescale = 10
        agent_mix = "us_income"
        bracket_setting = "three"
        percent_ego = 100  # ペルソナはこの設定のみサポートするため100%利己的を使用
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

    # OpenAI APIキーが設定されていることを確認
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ECON_OPENAI'):
        print("OPENAI_API_KEY環境変数を設定してください")
        return False

    try:
        run_simulation(args)
        print("✓ 限定合理性シミュレーションが正常に完了しました")
        return True
    except Exception as e:
        print(f"✗ 限定合理性シミュレーションが失敗しました: {e}")
        return False


def test_democratic_scenario():
    """民主的投票シナリオをテストする。"""
    print("民主的投票シナリオを実行中...")
    
    class Args:
        scenario = "democratic"
        num_agents = 3
        max_timesteps = 20
        worker_type = "LLM"
        planner_type = "LLM"
        llm = "gpt-4o-mini"
        port = 8000
        service = "vllm"
        use_openrouter = False
        prompt_algo = "io"
        history_len = 20
        timeout = 10
        two_timescale = 10
        agent_mix = "us_income"
        bracket_setting = "three"
        percent_ego = 100  # ペルソナはこの設定のみサポートするため100%利己的を使用
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

    # OpenAI APIキーが設定されていることを確認
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ECON_OPENAI'):
        print("OPENAI_API_KEY環境変数を設定してください")
        return False

    try:
        run_simulation(args)
        print("✓ 民主的シナリオのシミュレーションが正常に完了しました")
        return True
    except Exception as e:
        print(f"✗ 民主的シナリオのシミュレーションが失敗しました: {e}")
        return False


def test_fixed_workers():
    """固定ワーカーシナリオをテストする。"""
    print("固定ワーカーシナリオを実行中...")
    
    class Args:
        scenario = "rational"
        num_agents = 3
        max_timesteps = 20
        worker_type = "FIXED"
        planner_type = "FIXED"
        llm = "gpt-4o-mini"
        port = 8000
        service = "vllm"
        use_openrouter = False
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
    
    # OpenAI APIキーが設定されていることを確認
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ECON_OPENAI'):
        print("OPENAI_API_KEY環境変数を設定してください")
        return False

    try:
        run_simulation(args)
        print("✓ 固定ワーカーシミュレーションが正常に完了しました")
        return True
    except Exception as e:
        print(f"✗ 固定ワーカーシミュレーションが失敗しました: {e}")
        return False


def test_openrouter_rational():
    """OpenRouterを使用した合理的シナリオをテストする。"""
    print("OpenRouterで合理的シナリオを実行中...")
    
    class Args:
        scenario = "rational"
        num_agents = 3
        max_timesteps = 20
        worker_type = "LLM"
        planner_type = "LLM"
        llm = "meta-llama/llama-3.1-8b-instruct"
        port = 8000
        service = "vllm"
        use_openrouter = True
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
    
    # OpenRouter APIキーが設定されていることを確認
    if not os.getenv('OPENROUTER_API_KEY'):
        print("OPENROUTER_API_KEY環境変数を設定してください")
        print("例: export OPENROUTER_API_KEY=your_api_key_here")
        return False

    try:
        run_simulation(args)
        print("✓ OpenRouter合理的シミュレーションが正常に完了しました")
        return True
    except Exception as e:
        print(f"✗ OpenRouter合理的シミュレーションが失敗しました: {e}")
        return False


def test_vllm_rational():
    """ローカルvLLMサーバーを使用した合理的シナリオをテストする。"""
    print("ローカルvLLMで合理的シナリオを実行中...")
    
    class Args:
        scenario = "rational"
        num_agents = 3
        max_timesteps = 20
        worker_type = "LLM"
        planner_type = "LLM"
        llm = "meta-llama/Llama-3.1-8B-Instruct"
        port = 8000
        service = "vllm"
        use_openrouter = False
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
    
    print("vLLMサーバーがポート8000で起動していることを確認してください")
    print("例: python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8000")

    try:
        run_simulation(args)
        print("✓ vLLM合理的シミュレーションが正常に完了しました")
        return True
    except Exception as e:
        print(f"✗ vLLM合理的シミュレーションが失敗しました: {e}")
        return False


def test_ollama_rational():
    """Ollamaを使用した合理的シナリオをテストする。"""
    print("Ollamaで合理的シナリオを実行中...")
    
    class Args:
        scenario = "rational"
        num_agents = 3
        max_timesteps = 20
        worker_type = "LLM"
        planner_type = "LLM"
        llm = "llama3.1:8b"
        port = 11434
        service = "ollama"
        use_openrouter = False
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
    
    print("Ollamaがllama3.1:8bモデルで起動していることを確認してください")
    print("例: ollama run llama3.1:8b")

    try:
        run_simulation(args)
        print("✓ Ollama合理的シミュレーションが正常に完了しました")
        return True
    except Exception as e:
        print(f"✗ Ollama合理的シミュレーションが失敗しました: {e}")
        return False


def test_gemini_rational():
    """Google Geminiを使用した合理的シナリオをテストする。"""
    print("Google Geminiで合理的シナリオを実行中...")
    
    class Args:
        scenario = "rational"
        num_agents = 3
        max_timesteps = 20
        worker_type = "LLM"
        planner_type = "LLM"
        llm = "gemini-1.5-flash"
        port = 8000
        service = "vllm"
        use_openrouter = False
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
    
    # Google AI APIキーが設定されていることを確認
    if not os.getenv('GOOGLE_API_KEY'):
        print("GOOGLE_API_KEY環境変数を設定してください")
        print("例: export GOOGLE_API_KEY=your_api_key_here")
        return False

    try:
        run_simulation(args)
        print("✓ Gemini合理的シミュレーションが正常に完了しました")
        return True
    except Exception as e:
        print(f"✗ Gemini合理的シミュレーションが失敗しました: {e}")
        return False


def run_all_scenario_tests():
    """全てのシナリオテストを実行する。"""
    print("="*60)
    print("LLM Economist高度な使用例テストを実行中")
    print("全てのシミュレーションは20タイムステップで実行されます")
    print("="*60)

    # コアシナリオテスト（OpenAI APIキーが必要）
    core_tests = [
        ("合理的シナリオ", test_rational_openai),
        ("限定合理性", test_bounded_rationality),
        ("民主的投票", test_democratic_scenario),
        ("固定ワーカー", test_fixed_workers),
    ]

    # 追加サービステスト（各サービスのAPIキーが必要）
    service_tests = [
        ("OpenRouter", test_openrouter_rational),
        ("vLLM", test_vllm_rational),
        ("Ollama", test_ollama_rational),
        ("Gemini", test_gemini_rational),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    print("\n" + "="*40)
    print("コアシナリオテスト")
    print("="*40)
    
    for test_name, test_func in core_tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ テストが例外で失敗: {e}")
            failed += 1

    print("\n" + "="*40)
    print("追加サービステスト")
    print("="*40)
    
    for test_name, test_func in service_tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                skipped += 1  # APIキーが未設定の場合はスキップとしてカウント
        except Exception as e:
            print(f"✗ テストが例外で失敗: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"テスト結果: {passed} 成功, {failed} 失敗, {skipped} スキップ")
    print("="*60)

    if failed == 0:
        print("全ての利用可能なテストに合格しました!")
        if skipped > 0:
            print(f"注意: APIキーが未設定のため{skipped}件のテストがスキップされました")
    else:
        print("一部のテストが失敗しました。上記のエラーを確認してください。")
    
    return failed == 0


def main():
    """高度な使用例のメインエントリポイント。"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "--help" or command == "-h":
            print(__doc__)
            print("\n使い方:")
            print("  python examples/advanced_usage.py                    # 全テストを実行")
            print("  python examples/advanced_usage.py rational          # 合理的シナリオをテスト")
            print("  python examples/advanced_usage.py bounded           # 限定合理性をテスト")
            print("  python examples/advanced_usage.py democratic        # 民主的投票をテスト")
            print("  python examples/advanced_usage.py fixed             # 固定ワーカーをテスト")
            print("  python examples/advanced_usage.py openrouter        # OpenRouterをテスト")
            print("  python examples/advanced_usage.py vllm              # vLLMをテスト")
            print("  python examples/advanced_usage.py ollama            # Ollamaをテスト")
            print("  python examples/advanced_usage.py gemini            # Geminiをテスト")
            print("  python examples/advanced_usage.py --help            # このヘルプを表示")
            print("\n全てのシミュレーションはテスト目的で20タイムステップで実行されます。")
            return
        
        # 個別テストを実行
        test_map = {
            "rational": test_rational_openai,
            "bounded": test_bounded_rationality,
            "democratic": test_democratic_scenario,
            "fixed": test_fixed_workers,
            "openrouter": test_openrouter_rational,
            # "vllm": test_vllm_rational,
            # "ollama": test_ollama_rational,
            # "gemini": test_gemini_rational,
        }
        
        if command in test_map:
            success = test_map[command]()
            sys.exit(0 if success else 1)
        else:
            print(f"不明なコマンド: {command}")
            print("利用可能なコマンドを確認するには --help を使用してください")
            sys.exit(1)
    
    # 全テストを実行
    success = run_all_scenario_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 