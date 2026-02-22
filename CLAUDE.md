# CLAUDE.md

このファイルは、Claude Code（claude.ai/code）がこのリポジトリで作業する際のガイダンスを提供します。

## プロジェクト概要

LLM Economistは、LLMを経済エージェントとして活用し、税制最適化やメカニズムデザインを研究するためのマルチエージェント経済シミュレーションフレームワーク。Stackelbergゲーム（二層マルチエージェント強化学習）として実装されており、Tax Planner（リーダー）が社会厚生を最大化する税率を設定し、Worker（フォロワー）が税率に基づいて労働時間を最適化する。

## コマンド

```bash
# 環境構築
uv sync                   # 依存関係インストール
uv sync --dev             # dev依存含む（pytest, black, flake8, isort）

# テスト
uv run pytest -v                                    # 全テスト実行
uv run pytest tests/test_quickstart.py -v           # 基本機能テスト（APIキー不要）
uv run pytest tests/test_simulation.py -v           # シミュレーションロジック
uv run pytest tests/test_models.py -v               # LLMモデル統合テスト（APIキー必要）
uv run pytest tests/test_advanced_usage.py -v       # E2Eテスト（APIキー必要）
uv run pytest --cov=llm_economist --cov-report=html # カバレッジ付き

# シミュレーション実行
uv run python -m llm_economist.main --scenario rational --num-agents 5 --max-timesteps 500
uv run python -m llm_economist.main --scenario bounded --num-agents 10 --percent-ego 100
uv run python -m llm_economist.main --scenario democratic --num-agents 15 --two-timescale 50

# 論文実験の再現
uv run python experiments/run_experiments.py --experiment all
uv run python experiments/run_experiments.py --experiment rational

# フォーマット・リント
uv run black llm_economist/
uv run isort llm_economist/
uv run flake8 llm_economist/
```

## アーキテクチャ

### シミュレーションフロー（main.py）

`main()` → `run_simulation(args)` が全体の制御を行う：
1. LLM接続テスト（TestAgent）
2. スキル分布の初期化（GB2所得分布）
3. エージェント生成（ペルソナ生成含む）
4. タイムステップループ：Planner行動 → Worker並列行動 → 税適用 → 効用計算

### エージェント層 (`llm_economist/agents/`)

- **LLMAgent** (`llm_agent.py`): 全エージェントの基底クラス。メッセージ履歴管理、プロンプト戦略（IO/CoT/SC/MCTS）、JSON応答パース、税率パース・バリデーションを担当
- **Worker** (`worker.py`): 労働時間（0-100）を決定。3つの効用タイプ（egotistical/altruistic/adversarial）と3つのシナリオ（rational/bounded/democratic）をサポート。`distribute_personas()`でLLMベースのペルソナ生成
- **TaxPlanner** (`planner.py`): 限界税率を設定。社会厚生関数（SWF）を最適化。税率変更は DELTA ∈ [-20, -10, 0, 10, 20]%

### モデル層 (`llm_economist/models/`)

**BaseLLMModel** (`base.py`) を継承するプロバイダー実装：
- `openai_model.py`: OpenAI GPT

共通インターフェース: `send_msg(system_prompt, user_prompt, temperature, json_format) → (response, is_json)`

### ユーティリティ (`llm_economist/utils/`)

- `common.py`: GB2所得分布（`rGB2`/`qGB2`）、エージェント分配、投票集計、Saez最適税率計算
- `bracket.py`: 税率ブラケット設定（flat/three/US_FED）

## 環境変数

シミュレーション実行にはAPIキーが必要：
- `OPENAI_API_KEY` または `ECON_OPENAI`: OpenAI
- `WANDB_API_KEY`: 実験トラッキング（オプション）

## 重要なCLI引数

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--scenario` | rational / bounded / democratic | rational |
| `--num-agents` | ワーカーエージェント数 | 5 |
| `--llm` | 使用するLLMモデル名 | gpt-4o-mini |
| `--worker-type` | LLM / FIXED / ONE_LLM | LLM |
| `--planner-type` | LLM / US_FED / SAEZ / UNIFORM | LLM |
| `--prompt-algo` | io / cot | io |
| `--bracket-setting` | flat / three / US_FED | flat |
| `--use-multithreading` | Worker並列実行の有効化 | False |
| `--wandb` | W&Bロギングの有効化 | False |

## 拡張方法

LLMモデルは OpenAI API のみ使用する。カスタムモデルを追加する場合は `BaseLLMModel` を継承し `send_msg()` を実装する。カスタムWorkerは `Worker` クラスを継承し `compute_utility()` をオーバーライドする。
