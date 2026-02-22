# LLM Economist

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.15815-b31b1b.svg)](https://arxiv.org/abs/2507.15815)

<p align="center">
  <img src="fig/llm_econ_fig1.jpg" alt="LLM Economist Figure 1" width="600"/>
</p>

大規模言語モデル（LLM）を用いた経済シミュレーションのための包括的フレームワークです。LLM Economist は最先端の言語モデルを活用し、多様なエージェント集団によるリアルで動的な経済シミュレーションを構築することで、税制最適化やメカニズムデザインの研究を可能にします。

## 🚀 特徴

- **OpenAI API による LLM 統合**: OpenAI GPT モデルを活用した経済シミュレーション
- **多様な経済シナリオ**: 合理的エージェント、限定合理性、民主的投票メカニズム
- **リアルなエージェントペルソナ**: 実際の人口統計・職業データに基づく LLM 生成ペルソナ
- **スケーラブルなアーキテクチャ**: 効率的な並列処理により 3〜1000 以上のエージェントをサポート
- **包括的なテスト**: 実際の API 統合テストを含む完全なテストスイート
- **再現可能な研究**: 標準化された実験スクリプトと構成管理

## 📖 概要

LLM Economist フレームワークは、経済システムを二層のマルチエージェント強化学習問題としてモデル化し、シュタッケルベルグゲームとして実装しています：

1. **税制プランナー（リーダー）**: 社会的厚生を最大化するよう税制を設定
2. **労働者（フォロワー）**: 税制と個々の効用関数に基づいて労働配分を最適化

主な技術的革新：
- 合理的効用関数のための**インコンテキスト最適化**
- 実際の職業・年齢・性別統計を用いた**合成人口統計データ**によるリアルなエージェント多様性
- ユニークでリアルな経済エージェントを生成する **LLM 生成ペルソナ**
- 社会にポジティブな影響を与える**メカニズムデザイン**

## 🛠️ インストール

### 環境構築

```bash
git clone https://github.com/sethkarten/LLMEconomist.git
cd LLMEconomist
uv sync
```

### 開発用インストール

```bash
uv sync --dev
```

## 🚦 クイックスタート

### 1. API キーの設定

```bash
export OPENAI_API_KEY="your_openai_key"
```

### 2. 初めてのシミュレーション実行

```bash
# 合理的エージェントによるシンプルなシミュレーション
uv run python -m llm_economist.main --scenario rational --num-agents 5 --max-timesteps 500

# 限定合理性シミュレーション（注: 現在はペルソナ付きの100%利己的エージェントを使用）
uv run python -m llm_economist.main --scenario bounded --num-agents 10 --percent-ego 100

# 民主的投票シミュレーション
uv run python -m llm_economist.main --scenario democratic --num-agents 15 --two-timescale 50
```

### 3. 異なる LLM モデルを試す

```bash
# OpenAI GPT-4o
uv run python -m llm_economist.main --llm gpt-4o --scenario rational

# OpenAI GPT-4o-mini（コスト効率重視）
uv run python -m llm_economist.main --llm gpt-4o-mini --scenario rational
```

## 🏗️ プロジェクト構成

```
LLMEconomist/
├── llm_economist/              # メインパッケージ
│   ├── agents/                 # エージェント実装
│   │   ├── worker.py          # 労働者エージェントロジック
│   │   ├── planner.py         # 税制プランナーロジック
│   │   └── llm_agent.py       # LLM エージェント基底クラス
│   ├── models/                 # LLM モデル統合
│   │   ├── openai_model.py    # OpenAI GPT モデル
│   │   └── base.py            # モデル基底インターフェース
│   ├── utils/                  # ユーティリティ関数
│   │   ├── common.py          # 共通ユーティリティ
│   │   └── bracket.py         # 税率区分ユーティリティ
│   ├── data/                   # 人口統計データファイル
│   └── main.py                 # メインエントリーポイント
├── experiments/                # 実験スクリプト
├── examples/                   # 使用例
│   ├── quick_start.py         # 基本機能テスト
│   └── advanced_usage.py      # シミュレーションシナリオテスト
├── tests/                      # テストスイート
└── README.md                   # 本ファイル
```

## 🔧 設定オプション

### シミュレーションパラメータ

| パラメータ | 説明 | デフォルト値 | 選択肢 |
|-----------|------|------------|--------|
| `--scenario` | 経済シナリオ | `rational` | `rational`, `bounded`, `democratic` |
| `--num-agents` | 労働者エージェント数 | `5` | `1-1000+` |
| `--max-timesteps` | シミュレーション長 | `1000` | 任意の正の整数 |
| `--two-timescale` | 税制更新間のステップ数 | `25` | 任意の正の整数 |

### LLM 設定

| パラメータ | 説明 | デフォルト値 | 選択肢 |
|-----------|------|------------|--------|
| `--llm` | 使用する LLM モデル | `gpt-4o-mini` | `gpt-4o`, `gpt-4o-mini` |
| `--prompt-algo` | プロンプト戦略 | `io` | `io`, `cot` |

### エージェント設定

| パラメータ | 説明 | デフォルト値 | 選択肢 |
|-----------|------|------------|--------|
| `--worker-type` | 労働者エージェントタイプ | `LLM` | `LLM`, `FIXED` |
| `--planner-type` | プランナーエージェントタイプ | `LLM` | `LLM`, `US_FED`, `UNIFORM` |
| `--percent-ego` | 利己的エージェントの割合 (%) | `100` | `0-100` |
| `--percent-alt` | 利他的エージェントの割合 (%) | `0` | `0-100` |
| `--percent-adv` | 敵対的エージェントの割合 (%) | `0` | `0-100` |

**注意**: 現在、ペルソナ（`bounded` および `democratic` シナリオで使用）は利己的効用タイプのみをサポートしています。混合効用タイプはデフォルトペルソナでのみ使用可能です。

## 🤖 対応 LLM モデル

**OpenAI モデル:**
- `gpt-4o` - 最高性能、最高コスト
- `gpt-4o-mini` - 高速かつコスト効率に優れる（推奨）

## 📊 実験スクリプト

### 事前設定済み実験

論文の実験を実行します：

```bash
# 全実験
uv run python experiments/run_experiments.py --experiment all

# 個別実験
uv run python experiments/run_experiments.py --experiment rational
uv run python experiments/run_experiments.py --experiment bounded
uv run python experiments/run_experiments.py --experiment democratic
uv run python experiments/run_experiments.py --experiment llm_comparison
uv run python experiments/run_experiments.py --experiment scalability
```

### カスタム実験

```bash
# Chain of Thought プロンプティング
uv run python -m llm_economist.main --prompt-algo cot --llm gpt-4o

# Input-Output プロンプティング（デフォルト）
uv run python -m llm_economist.main --prompt-algo io --llm gpt-4o-mini

# 大規模シミュレーション
uv run python -m llm_economist.main --num-agents 100 --max-timesteps 2000
```

## 📈 使用例

本フレームワークは 2 種類の使用例を提供しています：

### 基本機能テスト

インポート、セットアップ、基本機能の簡易検証用：

```bash
# 全基本機能のテスト
uv run python examples/quick_start.py

# 個別の基本テストを実行
uv run python examples/quick_start.py --help
```

クイックスタートスクリプトが検証する内容：
- パッケージのインポートと依存関係
- 引数パーサーの設定
- API キーの検出
- 基本的な Args オブジェクトの作成
- サービス設定

### 上級使用例

20 タイムステップの実際のシミュレーションテスト用：

```bash
# 全シミュレーションシナリオの実行
uv run python examples/advanced_usage.py

# 個別シナリオのテスト
uv run python examples/advanced_usage.py rational          # OpenAI GPT-4o-mini
uv run python examples/advanced_usage.py bounded           # ペルソナ付き限定合理性
uv run python examples/advanced_usage.py democratic        # 民主的投票メカニズム
uv run python examples/advanced_usage.py fixed             # 固定労働者 + LLM プランナー

# 利用可能なシナリオの表示
uv run python examples/advanced_usage.py --help
```

全ての上級使用例は開発中の迅速な検証のため 20 タイムステップで実行されます。

### 使用例の構成

使用例は関心事の明確な分離を提供するよう整理されています：

- **`quick_start.py`**: シミュレーションを実行せずに基本機能を軽量に検証
  - インポートと依存関係のテスト
  - 設定セットアップの検証
  - API キーの利用可能性チェック
  - 高速実行（10 秒未満）

- **`advanced_usage.py`**: 実際の LLM API を使用したフルシミュレーションテスト
  - 20 タイムステップの経済シミュレーション
  - 全シナリオ: rational, bounded, democratic, fixed workers
  - OpenAI API による LLM 統合
  - 実践的なテスト（シナリオあたり 2〜10 分）

## 🧪 テスト

本フレームワークは 3 つのカテゴリに分類された包括的なテストを含んでいます：

### 基本機能テスト

```bash
# 基本機能のテスト（インポート、セットアップ、設定）
uv run pytest tests/test_quickstart.py -v

# 個別コンポーネントのテスト
uv run python examples/quick_start.py  # 基本機能の直接検証
```

### 統合テスト

```bash
# LLM モデル統合のテスト
uv run pytest tests/test_models.py -v

# モックを使用したシミュレーションロジックのテスト
uv run pytest tests/test_simulation.py -v

# 上級使用シナリオのテスト（API キーが必要）
uv run pytest tests/test_advanced_usage.py -v
```

### エンドツーエンドテスト

```bash
# 実際の API を使用したシミュレーションのテスト
uv run python examples/advanced_usage.py           # 全シナリオ
uv run python examples/advanced_usage.py rational  # 個別シナリオ
```

### フルテストスイート

```bash
# 全テストの実行
uv run pytest -v

# カバレッジ付きで実行
uv run pytest --cov=llm_economist --cov-report=html
```

### テスト要件

- **API キー**: 上級使用テストと統合テストには `OPENAI_API_KEY` または `ECON_OPENAI` が必要です
- **実際の統合**: 上級テストはエンドツーエンドの機能を確認するため実際の LLM API を使用します
- **高速実行**: 全テストは迅速な検証のため 20 タイムステップ以下で実行されます

## 🎭 エージェントペルソナ

本フレームワークは以下を使用してリアルなエージェントペルソナを生成します：

1. **人口統計サンプリング**: 国勢調査データに基づく実際の職業・年齢・性別統計
2. **LLM 生成**: サンプリングされた人口統計に基づき各ペルソナをユニークに生成
3. **経済的リアリズム**: リアルな所得水準、リスク許容度、生活環境を含むペルソナ

生成されたペルソナの例：
- *「あなたは55歳の女性で、准看護師として働いています... 30年以上の経験を持ち、退職後の貯蓄と医療ニーズを優先しています。」*
- *「あなたは53歳の男性で、溶接工として働いています... 退職後の貯蓄への懸念から、財政的に慎重な姿勢を取っています。」*

## 📚 研究の再現

LLM Economist 論文の実験を再現するには：

### セットアップ

1. **環境セットアップ:**
   ```bash
   git clone https://github.com/sethkarten/LLMEconomist.git
   cd LLMEconomist
   uv sync
   export WANDB_API_KEY="your_wandb_key"  # 実験追跡用
   ```

2. **LLM セットアップ:**
   ```bash
   export OPENAI_API_KEY="your_key"
   ```

### メイン実験

```bash
# 合理的エージェント
uv run python experiments/run_experiments.py --experiment rational --wandb

# 限定合理性
uv run python experiments/run_experiments.py --experiment bounded --wandb

# 民主的投票
uv run python experiments/run_experiments.py --experiment democratic --wandb

# LLM 比較
uv run python experiments/run_experiments.py --experiment llm_comparison --wandb

# スケーラビリティ分析
uv run python experiments/run_experiments.py --experiment scalability --wandb
```

## 🚀 上級機能

### カスタムエージェントタイプ

カスタムエージェント動作でフレームワークを拡張できます：

```python
from llm_economist.agents.worker import Worker

class CustomWorker(Worker):
    def compute_utility(self, income, rebate):
        # カスタム効用関数
        return your_custom_utility_logic(income, rebate)
```

### カスタム LLM モデル

`BaseLLMModel` を継承してカスタムモデルを実装できます：

```python
from llm_economist.models.base import BaseLLMModel

class CustomLLMModel(BaseLLMModel):
    def send_msg(self, system_prompt, user_prompt, temperature=None, json_format=False):
        # モデルの API を実装
        return response, is_json
```

### 実験追跡

Weights & Biases を使用した詳細な実験追跡を有効にできます：

```bash
uv run python -m llm_economist.main --wandb --scenario bounded --num-agents 20
```

## 🐛 トラブルシューティング

### よくある問題

**API キーエラー:**
```bash
# API キーが正しく設定されていることを確認
echo $OPENAI_API_KEY
```

**メモリの問題:**
- 大規模シミュレーションでは `--num-agents` を減らしてください
- コスト効率のため `gpt-4o` の代わりに `gpt-4o-mini` を使用してください
- メモリ使用量を削減するため `--history-len` を調整してください

**レート制限:**
- API コール間に遅延を追加してください
- コスト効率のため `gpt-4o-mini` の使用を検討してください

**テストの失敗:**
- API キーが設定されていることを確認してください
- ネットワーク接続を確認してください

## 📄 引用

本フレームワークを研究に使用する場合は、以下を引用してください：

```bibtex
@article{karten2025llm,
  title={LLM Economist: Large Population Models and Mechanism Design in Multi-Agent Generative Simulacra},
  author={Karten, Seth and Li, Wenzhe and Ding, Zihan and Kleiner, Samuel and Bai, Yu and Jin, Chi},
  journal={arXiv preprint arXiv:2507.15815},
  year={2025}
}
```

## 📝 ライセンス

本プロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。
