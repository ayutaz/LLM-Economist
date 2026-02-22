"""
LLM Economistフレームワークのメインエントリーポイント。
"""

import argparse
import logging
import os
import sys
import concurrent.futures
import torch.multiprocessing as mp
import wandb
import random
import numpy as np
import time
from .utils.common import distribute_agents, count_votes, rGB2, GEN_ROLE_MESSAGES
from .agents.worker import Worker, FixedWorker, distribute_personas
from .agents.llm_agent import TestAgent
from .agents.planner import TaxPlanner, FixedTaxPlanner


def setup_logging(args):
    """ロギング設定のセットアップ。"""
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    log_filename = f'{args.log_dir}/{args.name if args.name else "simulation"}.log'
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def run_simulation(args):
    """メインシミュレーションを実行する。"""
    logger = logging.getLogger('main')
    
    # LLM接続テスト
    if args.worker_type == 'LLM' or args.planner_type == 'LLM':
        try:
            TestAgent(args.llm, args.port, args)
            logger.info(f"Successfully connected to LLM: {args.llm}")
        except Exception as e:
            logger.error(f"Failed to connect to LLM: {e}")
            if args.worker_type == 'LLM' or args.planner_type == 'LLM':
                sys.exit(1)
    
    # スキル分布の初期化
    if args.agent_mix == 'uniform':
        skills = [-1] * args.num_agents  # worker.pyで一様分布にマッピング
    elif args.agent_mix == 'us_income':
        # 米国所得をスキルレベルに変換 (所得水準は週40時間労働時)
        skills = [float(x / 40) for x in rGB2(args.num_agents)] 
        print(skills)
        logger.info(f"Skills sampled from GB2 Distribution: {skills}")
    else:
        raise ValueError(f'不明なエージェントミックス: {args.agent_mix}')
    
    # エージェントの初期化
    agents = []
    personas = []
    
    if args.scenario == 'rational':
        personas = ['default' for i in range(args.num_agents)]
        utility_types = ['egotistical' for i in range(args.num_agents)]
    elif args.scenario in ('bounded', 'democratic'):
        # ペルソナの生成
        persona_data = distribute_personas(args.num_agents, args.llm, args.port, args.service)
        global GEN_ROLE_MESSAGES
        GEN_ROLE_MESSAGES.clear()
        GEN_ROLE_MESSAGES.update(persona_data)
        personas = list(GEN_ROLE_MESSAGES.keys())
        
        assert (args.percent_ego + args.percent_alt + args.percent_adv) == 100
        utility_types = distribute_agents(args.num_agents, [args.percent_ego, args.percent_alt, args.percent_adv])
        print('utility_types', utility_types)
        logger.info(f"Utility Types: {utility_types}")
    
    # ワーカーエージェントの作成
    for i in range(args.num_agents):
        name = f"worker_{i}"
        if args.worker_type == 'LLM' or (args.worker_type == 'ONE_LLM' and i == 0):
            agent = Worker(args.llm, 
                           args.port, 
                           name, 
                           utility_type=utility_types[i],
                           history_len=args.history_len, 
                           prompt_algo=args.prompt_algo, 
                           max_timesteps=args.max_timesteps, 
                           two_timescale=args.two_timescale,
                           role=personas[i], 
                           scenario=args.scenario, 
                           num_agents=args.num_agents,
                           args=args,
                           skill=skills[i],
                           )
        else:
            agent = FixedWorker(name, history_len=args.history_len, labor=np.random.randint(40, 61), args=args)
        agents.append(agent)
    
    # 税プランナーの初期化
    if args.planner_type == 'LLM':
        planner_history = args.history_len
        if args.num_agents > 20:
            planner_history = args.history_len//(args.num_agents) * 20
        
        tax_planner = TaxPlanner(args.llm, args.port, 'Joe', 
                                 history_len=planner_history, prompt_algo=args.prompt_algo, 
                                 max_timesteps=args.max_timesteps, num_agents=args.num_agents, args=args)
    elif args.planner_type in ['US_FED', 'SAEZ', 'SAEZ_FLAT', 'SAEZ_THREE', 'UNIFORM']:
        tax_planner = FixedTaxPlanner('Joe', args.planner_type, history_len=args.history_len, skills=skills, args=args)
    tax_rates = tax_planner.tax_rates
    
    # wandbロギングの初期化
    if args.wandb:
        experiment_name = generate_experiment_name(args)
        wandb.init(
            project="llm-economist",
            name=experiment_name,
            config=vars(args)
        )
    
    start_time = time.time()
    
    # メインシミュレーションループ
    for k in range(args.max_timesteps):
        logger.info(f"TIMESTEP {k}")
        print(f"TIMESTEP {k}")
        
        wandb_logger = {}
        
        # 新しい税率の取得
        workers_stats = [(agent.z, agent.utility) for agent in agents]
        # ウォームアップ期間中は税率を設定しない
        if k % args.two_timescale == 0 and args.planner_type == 'LLM' and k >= args.warmup:
            if args.scenario == 'democratic':
                # ThreadPoolExecutorを使用してエージェントアクションを並列実行
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_agents) as executor:
                    if args.platforms:
                        max_retries = 10
                        retry_count = 0
                        candidates = []
                        while not candidates and retry_count < max_retries:
                            futures0 = [executor.submit(agent.act_pre_vote, k) for agent in agents]
                            concurrent.futures.wait(futures0)
                            candidates = [(agent.name.split("_")[-1], agent.platform) for agent in agents if agent.platform != None]
                            retry_count += 1  # 無限ループの防止
                        logger.info(f"Candidates: {candidates}")
                        print("Candidates: ", candidates)
                        if not candidates:
                            print("候補者なし。全エージェントの投票は更新されず、現在のリーダーが政権を維持します")
                            logger.info(f"候補者なし。全エージェントの投票は更新されず、現在のリーダーが政権を維持します")
                        else:
                            futures = [executor.submit(agent.act_vote_platform, candidates, k) for agent in agents]
                            concurrent.futures.wait(futures)
                    else:
                        futures = [executor.submit(agent.act_vote, k) for agent in agents]
                        concurrent.futures.wait(futures)
                votes_list = [agent.vote for agent in agents]
                print("Votes: ", votes_list)
                leader_agent = count_votes(votes_list)
                leader = agents[leader_agent] # 不要
                wandb_logger[f"leader"] = leader_agent
                print("leader: ", leader_agent)
                if args.platforms:
                    for agent in agents:
                        agent.update_leader(k, leader_agent, candidates)
                    tax_planner.update_leader(k, leader_agent, candidates)
                else:
                    for agent in agents:
                        agent.update_leader(k, leader_agent)
                    tax_planner.update_leader(k, leader_agent)
                # 税率の取得
                # エージェント特徴量のみのプランナーメッセージを取得
                planner_state = tax_planner.get_state(k, workers_stats, True)
                # 税率を決定
                tax_delta = agents[leader_agent].act_plan(k, planner_state)[0]
                print("act_leader: ", tax_delta)
                for agent in agents:
                    agent.update_leader_action(k, tax_delta)
                tax_planner.update_leader_action(k, tax_delta) # act_log_onlyとの併用では不要かもしれない
                tax_rates = tax_planner.act_log_only(tax_delta, k)
                for agent in agents:
                    agent.tax_rates = tax_rates
            else:
                tax_rates = tax_planner.act(k, workers_stats)
                print("act: ", tax_rates)
        elif args.planner_type == 'LLM':
            tax_planner.add_obs_msg(k, workers_stats)
            tax_planner.add_act_msg(k, tax_rates=tax_rates)

        planner_state = None
        if args.percent_ego < 100:
            planner_state = tax_planner.get_state(k, workers_stats, False) # 敵対的・利他的エージェント用
            
        if args.use_multithreading:
            # ThreadPoolExecutorを使用してエージェントアクションを並列実行
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_agents) as executor:
                futures = [executor.submit(agent.act, k, tax_rates, planner_state) for agent in agents]
                concurrent.futures.wait(futures)
        else:
            for i in range(args.num_agents):
                agents[i].act(k, tax_rates, planner_state)

        pre_tax_incomes = [agents[i].z for i in range(args.num_agents)]
        
        # 税金の計算
        post_tax_incomes, total_tax = tax_planner.apply_taxes(tax_rates, pre_tax_incomes)
        tax_indv = np.array(pre_tax_incomes) - np.array(post_tax_incomes)
        tax_rebate_avg = total_tax / args.num_agents
        
        # エージェントの効用を更新
        for i, agent in enumerate(agents):
            agent.tax_paid = tax_indv[i]
        if args.scenario == 'bounded' and args.use_multithreading:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_agents) as executor:
                futures = [executor.submit(agents[i].update_utility, k, post_tax_incomes[i], tax_rebate_avg, tax_planner.swf) for i in range(args.num_agents)]
                concurrent.futures.wait(futures)
        else:
            for i, agent in enumerate(agents):
                agent.update_utility(k, post_tax_incomes[i], tax_rebate_avg, tax_planner.swf)
        for i, agent in enumerate(agents):
            agent.log_stats(k, wandb_logger, debug=args.debug)
        
        # 税プランナーの統計をログ
        # 利他的/敵対的プランナーのSWFにはデフォルトで等弾力性効用を使用
        u = [agents[i].utility if agents[i].utility_type == 'egotistical' else agents[i].compute_isoelastic_utility(post_tax_incomes[i], tax_rebate_avg) for i in range(args.num_agents)]
        tax_planner.log_stats(k, wandb_logger, z=pre_tax_incomes, u=u, debug=args.debug)
        
        if args.wandb:
            wandb.log(wandb_logger)
            
        end_time = time.time()
        iteration_time = end_time - start_time
        total_actions = (k + 1) * args.num_agents  # これまでに実行された合計アクション数

        fps = (k + 1) / iteration_time
        aps = total_actions / iteration_time

        logger.info(f"Time for iteration 0-{k+1}: {iteration_time:.5f} seconds")
        logger.info(f"FPS: {fps:.2f}")
        logger.info(f"APS: {aps:.2f}")

        remaining_time = (args.max_timesteps - k - 1) * iteration_time / (k + 1)
        logger.info(f"Time remaining {k+2}-{args.max_timesteps}: {remaining_time:.5f} seconds")

    logger.info("シミュレーションが正常に完了しました!")
    
    if args.wandb:
        wandb.finish()


def generate_experiment_name(args):
    """説明的な実験名を生成する。"""
    # シナリオとエージェント数をベースとして開始
    name_parts = [f"{args.scenario}"]
    name_parts.append(f"a{args.num_agents}")
    
    # 全員が利己的でない場合、エージェント構成を追加
    if args.percent_ego != 100:
        name_parts.append(f"mix_e{args.percent_ego}_a{args.percent_alt}_d{args.percent_adv}")
    
    # ワーカーとプランナーのタイプを追加
    name_parts.append(f"w-{args.worker_type}")
    name_parts.append(f"p-{args.planner_type}")
    
    # LLMモデル名を追加 (短縮)
    llm_name = args.llm.replace("llama3:", "l3-").replace("gpt-", "g").replace("-mini-2024-07-18", "m")
    name_parts.append(f"llm-{llm_name}")
    
    # プロンプティングアルゴリズムを追加
    name_parts.append(f"prompt-{args.prompt_algo}")
    
    # タイムスケールと履歴長を追加
    name_parts.append(f"ts{args.two_timescale}")
    name_parts.append(f"hist{args.history_len}")
    
    # 最大タイムステップ数を追加
    name_parts.append(f"steps{args.max_timesteps}")
    
    # ブラケット設定を追加
    name_parts.append(f"bracket-{args.bracket_setting}")
    
    # プラットフォーム使用時に投票インジケーターを追加
    if args.platforms:
        name_parts.append("voting")
    
    # 全パーツをアンダースコアで結合
    return "_".join(name_parts)


def create_argument_parser():
    """引数パーサーを作成して返す。"""
    parser = argparse.ArgumentParser(description='AI Economistシミュレーション')
    parser.add_argument('--num-agents', type=int, default=5, help='シミュレーション内のエージェント数')
    parser.add_argument('--worker-type', default='LLM', choices=['LLM', 'FIXED', 'ONE_LLM'], help='ワーカーエージェントのタイプ')
    parser.add_argument('--planner-type', default='LLM', choices=['LLM', 'US_FED', 'SAEZ', 'SAEZ_THREE', 'SAEZ_FLAT', 'UNIFORM'], help='税プランナーのタイプ')
    parser.add_argument('--max-timesteps', type=int, default=1000, help='シミュレーションの最大タイムステップ数')
    parser.add_argument('--history-len', type=int, default=50, help='考慮する履歴の長さ')
    parser.add_argument('--two-timescale', type=int, default=25, help='2タイムスケール更新の間隔')
    parser.add_argument('--debug', type=bool, default=True, help='デバッグモードを有効化')
    parser.add_argument('--llm', default='llama3:8b', type=str, help='使用する言語モデル')
    parser.add_argument('--prompt-algo', default='io', choices=['io', 'cot'], help='使用するプロンプティングアルゴリズム')
    parser.add_argument('--scenario', default='rational', choices=['rational', 'bounded', 'democratic'], help='シナリオ')
    parser.add_argument('--percent-ego', type=int, default=100)
    parser.add_argument('--percent-alt', type=int, default=0)
    parser.add_argument('--percent-adv', type=int, default=0)
    parser.add_argument('--port', type=int, default=8009)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--agent-mix', default='us_income', choices=['uniform', 'us_income'], help='エージェントのスキルレベルの分布')
    parser.add_argument('--platforms', action="store_true", help='エージェントが選挙でプラットフォームを掲げて立候補する')
    parser.add_argument('--name', type=str, default='', help='実験名')
    parser.add_argument('--log-dir', type=str, default='logs', help='ログファイルのディレクトリ')
    parser.add_argument('--bracket-setting', default='three', choices=['flat', 'three', 'US_FED'])
    parser.add_argument('--service', default='vllm', choices=['vllm', 'ollama'])
    parser.add_argument('--use-multithreading', action='store_true')
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--elasticity', nargs='+', type=float, default=[0.4],
                    help='税ブラケットの弾力性値')
    parser.add_argument('--wandb', action='store_true', help='wandbロギングを有効化')
    parser.add_argument('--timeout', type=int, default=30, help='LLM呼び出しのタイムアウト')

    return parser


def main():
    """メインエントリーポイント。"""
    parser = create_argument_parser()
    args = parser.parse_args()

    if not args.name:
        args.name = generate_experiment_name(args)

    # ロギングのセットアップ
    setup_logging(args)
    
    # メインプロセスでロガーを作成
    logger = logging.getLogger('main')
    pid = os.getpid()

    logger.info(f"Main process started: {args.name}")
    logger.info(f'PID: {pid}')
    logger.info(args)
    
    # 乱数シードの設定
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    start_time = time.time()
    
    run_simulation(args)
    
    end_time = time.time()
    print(f"シミュレーション総時間: {end_time - start_time:.2f} 秒")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main() 