from enum import Enum
from collections import Counter
import random
import os
import numpy as np
from scipy import stats
import pandas as pd
from typing import List, Tuple

KEY = os.getenv('ECON_OPENAI')

class Message(Enum):
    SYSTEM = 1
    UPDATE = 2
    ACTION = 3

# ペルソナメッセージの共有状態
GEN_ROLE_MESSAGES = {}

def labor_list(num_agents):
    base_value = 50
    offset = 10
    
    # エージェント数が増えるにつれて開始値が減少
    start_value = base_value - (num_agents - 1) * offset // 2
    
    # リストの生成
    return [abs(start_value + i * offset) % 100 for i in range(num_agents)]

def count_votes(votes_list: list):
    # 各候補者の得票数をカウント
    max_count = max(votes_list.count(vote) for vote in set(votes_list))

    # 最大得票数の候補者を全て取得
    tied_candidates = [vote for vote in set(votes_list) if votes_list.count(vote) == max_count]

    # 同票の候補者からランダムに当選者を選択
    elected_tax_planner = random.choice(tied_candidates)

    # 当選者名から整数インデックスを抽出
    #elected_tax_planner = int(winner.split("_")[-1])

    return elected_tax_planner

def distribute_agents(num_agents, agent_mix):
    # 各グループのエージェント数を概算
    adversarial_agents = round(agent_mix[2] / 100 * num_agents)
    selfless_agents = round(agent_mix[1] / 100 * num_agents)
    greedy_agents = num_agents - adversarial_agents - selfless_agents  # Remaining agents go to greedy

    # エージェントタイプのリストを返す
    agents = ['adversarial'] * adversarial_agents + ['altruistic'] * selfless_agents + ['egotistical'] * greedy_agents

    # リストをシャッフルしてエージェントの割り当てをランダム化
    random.shuffle(agents)

    return agents

# GAMLSSパッケージのRソースコードに準拠
# デフォルトはACS 2023の米国所得分布
def qGB2(p, mu=72402.78177917618, sigma=2.0721070746154746, nu=0.48651871959386955, tau=1.1410398548220329, lower_tail=True, log_p=False):
    """
    第2種一般化ベータ分布 (GB2) の分位関数。

    Parameters:
    -----------
    p : float or array-like
        確率
    mu : float
        スケールパラメータ (正の値)
    sigma : float
        形状パラメータ
    nu : float
        形状パラメータ (正の値)
    tau : float
        形状パラメータ (正の値)
    lower_tail : bool
        Trueの場合、確率はP[X <= x]、それ以外はP[X > x]
    log_p : bool
        Trueの場合、確率はlog(p)で与えられる

    Returns:
    --------
    q : float or array-like
        pの確率に対応する分位数
    """
    # パラメータバリデーション
    if np.any(mu <= 0):
        raise ValueError("muは正の値でなければなりません")
    if np.any(nu <= 0):
        raise ValueError("nuは正の値でなければなりません")
    if np.any(tau <= 0):
        raise ValueError("tauは正の値でなければなりません")

    # 必要に応じてlog確率を処理
    if log_p:
        p = np.exp(p)
    
    # 確率の範囲をバリデーション
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("pは0と1の間でなければなりません")
    
    # lower.tailパラメータの処理
    if not lower_tail:
        p = 1 - p
    
    # sigmaの符号処理
    if hasattr(sigma, "__len__"):
        p = np.where(sigma < 0, 1 - p, p)
    else:
        if sigma < 0:
            p = 1 - p
    
    # F分布の分位関数を使用 (ppfはRのqfに対応するscipyの関数)
    w = stats.f.ppf(p, 2 * nu, 2 * tau)
    
    # GB2分位数に変換
    q = mu * (((nu/tau) * w)**(1/sigma))
    
    return q

# デフォルトはACS 2023の米国所得分布
def rGB2(n, mu=72402.78177917618, sigma=2.0721070746154746, nu=0.48651871959386955, tau=1.1410398548220329):
    """
    第2種一般化ベータ分布 (GB2) からのランダムサンプルを生成する。

    Parameters:
    -----------
    n : int
        生成するランダム値の数
    mu : float
        スケールパラメータ (正の値)
    sigma : float
        形状パラメータ
    nu : float
        形状パラメータ (正の値)
    tau : float
        形状パラメータ (正の値)

    Returns:
    --------
    r : array-like
        GB2分布からのランダムサンプル
    """
    # パラメータバリデーション
    if np.any(mu <= 0):
        raise ValueError("muは正の値でなければなりません")
    if np.any(nu <= 0):
        raise ValueError("nuは正の値でなければなりません")
    if np.any(tau <= 0):
        raise ValueError("tauは正の値でなければなりません")

    # nを整数に変換
    n = int(np.ceil(n))
    
    # 一様乱数を生成
    p = np.random.uniform(0, 1, size=n)
    
    # 分位関数を使用して変換
    r = qGB2(p, mu=mu, sigma=sigma, nu=nu, tau=tau)
    
    return r

def linear_transform(samples, old_min, old_max, new_min, new_max):
    """NumPyを使用した効率的な線形変換"""
    samples_array = np.array(samples)
    transformed = (samples_array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return transformed

import numpy as np
from scipy import stats

def saez_optimal_tax_rates(skills, brackets, elasticities):
    """
    スキルに基づいて所得ブラケットのSaez最適限界税率を計算する。

    Parameters:
    -----------
    skills : list of float
        個人のスキルリスト (所得/100)。
    brackets : list of float
        所得カットオフポイントのリスト [min1, min2, ..., max_value];
        連続するペアが1つのブラケットを定義する。
    elasticities : float or list of float
        単一のfloatの場合: 全ブラケットにこの弾力性を適用。
        リストの場合: 長さ = (ブラケット数)、つまりlen(brackets)-1、
        ブラケットごとに1つの弾力性を指定。

    Returns:
    --------
    tax_rates : list of float
        各ブラケットの最適限界税率 (パーセント)
        (例: [12.88, 3.23, 3.23])。
    """
    # スキルを所得に変換
    incomes = np.array(skills) * 100.0
    brackets = np.array(brackets)
    
    # 弾力性リストの構築
    n_brackets = len(brackets) - 1
    if isinstance(elasticities, (int, float)):
        elasticities = [float(elasticities)] * n_brackets
    else:
        if len(elasticities) != n_brackets:
            raise ValueError(f"elasticitiesの長さは{n_brackets}でなければなりませんが、{len(elasticities)}が指定されました")
        elasticities = [float(e) for e in elasticities]
    
    # 所得をソートし厚生ウェイトを計算
    incomes = np.sort(incomes)
    welfare_weights = 1.0 / np.maximum(incomes, 1e-10)
    welfare_weights /= welfare_weights.sum()
    
    # 密度を推定
    kde = stats.gaussian_kde(incomes)
    
    tax_rates = []
    for i in range(n_brackets):
        bracket_start, bracket_end = brackets[i], brackets[i+1]
        # zを中間点で選択 (最上位ブラケットの場合は開始点付近)
        if i < n_brackets - 1:
            z = 0.5 * (bracket_start + bracket_end)
        else:
            z = bracket_start + 0.1 * (bracket_end - bracket_start)
        
        F_z = np.mean(incomes <= z)
        f_z = kde(z)[0]
        
        # パレートテールパラメータ a(z)
        if F_z < 1.0:
            a_z = (z * f_z) / (1.0 - F_z)
        else:
            a_z = 10.0
        
        # 最上位ブラケットではa(z)を精緻化
        incomes_above = incomes[incomes >= z]
        if i == n_brackets - 1 and incomes_above.size > 0:
            m = incomes_above.mean()
            a_z = m / (m - bracket_start)
        
        # G(z): average welfare weight above z, normalized
        if incomes_above.size > 0 and F_z < 1.0:
            G_z = welfare_weights[incomes >= z].sum() / (1.0 - F_z)
        else:
            G_z = 0.0
        
        # このブラケットに適切な弾力性を選択
        ε = elasticities[i]
        
        # Saez最適税率 τ = (1 - G) / [1 - G + a * ε]
        tau = (1.0 - G_z) / (1.0 - G_z + a_z * ε)
        tau = max(0.0, min(1.0, tau))
        
        tax_rates.append(round(tau * 100, 2))
    
    return tax_rates

def generate_synthetic_data(csv_path: str, n_samples: int) -> List[Tuple[str, str, int]]:
    """
    職業・性別・年齢の分布に従って合成データポイントを生成する。

    Args:
        csv_path: 分布データを含むCSVファイルのパス
        n_samples: 生成する合成データポイントの数

    Returns:
        各要素が (職業, 性別, 年齢) のタプルのリスト
    """
    # CSVファイルの読み込み
    df = pd.read_csv(csv_path)
    
    # 年齢カテゴリラベルと対応する年齢範囲を作成
    age_columns = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
    age_ranges = {
        'Under 18': (14, 17),  # 就労可能年齢を14歳からと仮定
        '18-24': (18, 24),
        '25-34': (25, 34),
        '35-44': (35, 44),
        '45-54': (45, 54),
        '55-64': (55, 64),
        '65-74': (65, 74),
        '75+': (75, 90)  # 最高就労年齢を90歳と仮定
    }
    
    # 合計分布を計算
    total_distribution = df[age_columns].sum().sum()
    
    # 合成データを格納するリストを作成
    synthetic_data = []
    
    for _ in range(n_samples):
        # 分布に基づいてランダムに行を選択
        random_value = random.uniform(0, total_distribution)
        cumulative_sum = 0
        selected_row = None
        selected_age_column = None
        
        for idx, row in df.iterrows():
            for age_col in age_columns:
                cumulative_sum += row[age_col]
                if cumulative_sum >= random_value:
                    selected_row = row
                    selected_age_column = age_col
                    break
            if selected_row is not None:
                break
        
        # 選択された職業、性別を取得し、範囲内の具体的な年齢を生成
        occupation = selected_row['Occupation_Label']
        sex = selected_row['SEX_Label']
        age_range = age_ranges[selected_age_column]
        specific_age = random.randint(age_range[0], age_range[1])
        
        # 合成データポイントを追加
        synthetic_data.append((occupation, sex, specific_age))
    
    return synthetic_data