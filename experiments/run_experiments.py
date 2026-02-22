"""
LLM Economist 論文の主要実験を実行するスクリプト。
"""

import os
import sys
import subprocess
import argparse
from typing import List, Dict, Any


def run_command(cmd: List[str], description: str = ""):
    """コマンドを実行し、エラーを処理する。"""
    print(f"実行中: {description}")
    print(f"コマンド: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"成功: {description}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"{description} の実行中にエラー: {e}")
        print(f"標準エラー出力: {e.stderr}")
        return None


def rational_agents_experiment(args):
    """合理的エージェント実験を実行する。"""
    base_cmd = [
        sys.executable, "-m", "llm_economist.main",
        "--scenario", "rational",
        "--num-agents", str(args.num_agents),
        "--worker-type", "LLM",
        "--planner-type", "LLM",
        "--max-timesteps", str(args.max_timesteps),
        "--history-len", str(args.history_len),
        "--two-timescale", str(args.two_timescale),
        "--prompt-algo", args.prompt_algo,
        "--llm", args.llm
    ]
    
    if args.wandb:
        base_cmd.append("--wandb")
    
    if args.port:
        base_cmd.extend(["--port", str(args.port)])
    
    if args.service:
        base_cmd.extend(["--service", args.service])
    
    return run_command(base_cmd, "Rational Agents Experiment")


def bounded_rational_experiment(args):
    """限定合理性エージェント実験を実行する。"""
    base_cmd = [
        sys.executable, "-m", "llm_economist.main",
        "--scenario", "bounded",
        "--num-agents", str(args.num_agents),
        "--worker-type", "LLM",
        "--planner-type", "LLM",
        "--max-timesteps", str(args.max_timesteps),
        "--history-len", str(args.history_len),
        "--two-timescale", str(args.two_timescale),
        "--prompt-algo", args.prompt_algo,
        "--llm", args.llm,
        "--percent-ego", str(args.percent_ego),
        "--percent-alt", str(args.percent_alt),
        "--percent-adv", str(args.percent_adv)
    ]
    
    if args.wandb:
        base_cmd.append("--wandb")
    
    if args.port:
        base_cmd.extend(["--port", str(args.port)])
    
    if args.service:
        base_cmd.extend(["--service", args.service])
    
    return run_command(base_cmd, "Bounded Rational Agents Experiment")


def democratic_voting_experiment(args):
    """民主的投票実験を実行する。"""
    base_cmd = [
        sys.executable, "-m", "llm_economist.main",
        "--scenario", "democratic",
        "--num-agents", str(args.num_agents),
        "--worker-type", "LLM",
        "--planner-type", "LLM",
        "--max-timesteps", str(args.max_timesteps),
        "--history-len", str(args.history_len),
        "--two-timescale", str(args.two_timescale),
        "--prompt-algo", args.prompt_algo,
        "--llm", args.llm
    ]
    
    if args.wandb:
        base_cmd.append("--wandb")
    
    if args.port:
        base_cmd.extend(["--port", str(args.port)])
    
    if args.service:
        base_cmd.extend(["--service", args.service])
    
    return run_command(base_cmd, "Democratic Voting Experiment")


def llm_comparison_experiment(args):
    """LLM比較実験を実行する。"""
    models = ["gpt-4o-mini", "llama3:8b", "meta-llama/llama-3.1-8b-instruct"]
    
    for model in models:
        base_cmd = [
            sys.executable, "-m", "llm_economist.main",
            "--scenario", "rational",
            "--num-agents", str(args.num_agents),
            "--worker-type", "LLM",
            "--planner-type", "LLM",
            "--max-timesteps", str(args.max_timesteps),
            "--history-len", str(args.history_len),
            "--two-timescale", str(args.two_timescale),
            "--prompt-algo", args.prompt_algo,
            "--llm", model
        ]
        
        if args.wandb:
            base_cmd.append("--wandb")
        
        if args.port and "llama" in model:
            base_cmd.extend(["--port", str(args.port)])
        
        if args.service and "llama" in model:
            base_cmd.extend(["--service", args.service])
        
        run_command(base_cmd, f"LLM Comparison - {model}")


def scalability_experiment(args):
    """異なるエージェント数でスケーラビリティ実験を実行する。"""
    agent_counts = [5, 10, 25, 50, 100]
    
    for num_agents in agent_counts:
        base_cmd = [
            sys.executable, "-m", "llm_economist.main",
            "--scenario", "rational",
            "--num-agents", str(num_agents),
            "--worker-type", "LLM",
            "--planner-type", "LLM",
            "--max-timesteps", str(args.max_timesteps),
            "--history-len", str(args.history_len),
            "--two-timescale", str(args.two_timescale),
            "--prompt-algo", args.prompt_algo,
            "--llm", args.llm
        ]
        
        if args.wandb:
            base_cmd.append("--wandb")
        
        if args.port:
            base_cmd.extend(["--port", str(args.port)])
        
        if args.service:
            base_cmd.extend(["--service", args.service])
        
        run_command(base_cmd, f"Scalability Test - {num_agents} agents")



def tax_year_ablation_experiment(args):
    """課税年度の長さに関するアブレーション実験を実行する。"""
    timescales = [5, 10, 25, 50, 100]
    
    for timescale in timescales:
        # 同程度の課税年度数になるよう max_timesteps を調整
        max_timesteps = timescale * 20  # 20課税年度
        
        base_cmd = [
            sys.executable, "-m", "llm_economist.main",
            "--scenario", "rational",
            "--num-agents", str(args.num_agents),
            "--worker-type", "LLM",
            "--planner-type", "LLM",
            "--max-timesteps", str(max_timesteps),
            "--history-len", str(args.history_len),
            "--two-timescale", str(timescale),
            "--prompt-algo", args.prompt_algo,
            "--llm", args.llm
        ]
        
        if args.wandb:
            base_cmd.append("--wandb")
        
        if args.port:
            base_cmd.extend(["--port", str(args.port)])
        
        if args.service:
            base_cmd.extend(["--service", args.service])
        
        run_command(base_cmd, f"Tax Year Ablation - {timescale} steps")


def main():
    """メインの実験実行関数。"""
    parser = argparse.ArgumentParser(description="LLM Economist の実験を実行する")

    # 実験の選択
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["rational", "bounded", "democratic",
                                "llm_comparison", "scalability",
                                "tax_year_ablation", "all"],
                        help="実行する実験")

    # 共通パラメータ
    parser.add_argument("--num-agents", type=int, default=5,
                        help="エージェント数")
    parser.add_argument("--max-timesteps", type=int, default=2500,
                        help="最大タイムステップ数")
    parser.add_argument("--history-len", type=int, default=50,
                        help="履歴の長さ")
    parser.add_argument("--two-timescale", type=int, default=25,
                        help="2タイムスケールパラメータ")
    parser.add_argument("--prompt-algo", type=str, default="io",
                        choices=["io", "cot"],
                        help="プロンプトアルゴリズム")
    parser.add_argument("--llm", type=str, default="gpt-4o-mini",
                        help="使用するLLMモデル")
    parser.add_argument("--port", type=int, default=8000,
                        help="ローカルLLMサーバーのポート")
    parser.add_argument("--service", type=str, default="vllm",
                        choices=["vllm", "ollama"],
                        help="ローカルLLMサービス")

    # 限定合理性パラメータ
    parser.add_argument("--percent-ego", type=int, default=100,
                        help="利己的エージェントの割合")
    parser.add_argument("--percent-alt", type=int, default=0,
                        help="利他的エージェントの割合")
    parser.add_argument("--percent-adv", type=int, default=0,
                        help="敵対的エージェントの割合")

    # ログ
    parser.add_argument("--wandb", action="store_true",
                        help="WandBログを有効にする")
    
    args = parser.parse_args()
    
    # 選択された実験を実行
    if args.experiment == "rational" or args.experiment == "all":
        rational_agents_experiment(args)
    
    if args.experiment == "bounded" or args.experiment == "all":
        bounded_rational_experiment(args)
    
    if args.experiment == "democratic" or args.experiment == "all":
        democratic_voting_experiment(args)
    
    if args.experiment == "llm_comparison" or args.experiment == "all":
        llm_comparison_experiment(args)
    
    if args.experiment == "scalability" or args.experiment == "all":
        scalability_experiment(args)
    

    
    if args.experiment == "tax_year_ablation" or args.experiment == "all":
        tax_year_ablation_experiment(args)
    
    print("全実験が完了しました!")


if __name__ == "__main__":
    main() 