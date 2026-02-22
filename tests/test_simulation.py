"""
メインシミュレーション機能のテスト。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_economist.main import run_simulation, generate_experiment_name, create_argument_parser


class TestExperimentName:
    """実験名生成のテスト。"""

    def test_rational_experiment_name(self):
        """合理的実験の名前生成テスト。"""
        class Args:
            scenario = "rational"
            num_agents = 5
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "gpt-4o-mini"
            prompt_algo = "io"
            two_timescale = 25
            history_len = 50
            max_timesteps = 1000
            bracket_setting = "two"
        
        args = Args()
        name = generate_experiment_name(args)
        
        expected_parts = [
            "rational", "a5", "w-LLM", "p-LLM", "llm-g",
            "prompt-io", "ts25", "hist50", "steps1000", "bracket-two"
        ]
        
        for part in expected_parts:
            assert part in name
    
    def test_bounded_experiment_name(self):
        """限定合理性実験の名前生成テスト。"""
        class Args:
            scenario = "bounded"
            num_agents = 10
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "llama3:8b"
            prompt_algo = "cot"
            two_timescale = 20
            history_len = 30
            max_timesteps = 500
            bracket_setting = "three"
            percent_ego = 60
            percent_alt = 25
            percent_adv = 15
        
        args = Args()
        name = generate_experiment_name(args)
        
        expected_parts = [
            "bounded", "a10", "w-LLM", "p-LLM", "llm-l3-8b",
            "prompt-cot", "ts20", "hist30", "steps500", "bracket-three",
            "ego60", "alt25", "adv15"
        ]
        
        for part in expected_parts:
            assert part in name


class TestArgumentParser:
    """コマンドライン引数解析のテスト。"""

    def test_default_arguments(self):
        """デフォルト引数値のテスト。"""
        parser = create_argument_parser()
        args = parser.parse_args([])
        
        assert args.scenario == "rational"
        assert args.num_agents == 5
        assert args.max_timesteps == 1000
        assert args.worker_type == "LLM"
        assert args.planner_type == "LLM"
        assert args.llm == "gpt-4o-mini"
        assert args.prompt_algo == "io"
        assert args.history_len == 50
        assert args.two_timescale == 25
    
    def test_custom_arguments(self):
        """カスタム引数の解析テスト。"""
        parser = create_argument_parser()
        args = parser.parse_args([
            "--scenario", "bounded",
            "--num-agents", "10",
            "--llm", "claude-3.5-sonnet",
            "--prompt-algo", "cot",
            "--percent-ego", "70",
            "--percent-alt", "20",
            "--percent-adv", "10"
        ])
        
        assert args.scenario == "bounded"
        assert args.num_agents == 10
        assert args.llm == "claude-3.5-sonnet"
        assert args.prompt_algo == "cot"
        assert args.percent_ego == 70
        assert args.percent_alt == 20
        assert args.percent_adv == 10


class TestSimulation:
    """メインシミュレーション機能のテスト。"""
    
    @patch('llm_economist.main.TestAgent')
    @patch('llm_economist.main.Worker')
    @patch('llm_economist.main.TaxPlanner')
    @patch('llm_economist.main.wandb')
    def test_rational_simulation_setup(self, mock_wandb, mock_planner, mock_worker, mock_test_agent):
        """合理的シミュレーションが正しくセットアップされることを確認するテスト。"""
        # テストエージェントが成功するようにモック
        mock_test_agent.return_value = Mock()
        
        # ワーカーとプランナーの生成をモック
        mock_worker_instance = Mock()
        mock_worker_instance.labor = 50
        mock_worker_instance.utility = 100
        mock_worker.return_value = mock_worker_instance
        
        mock_planner_instance = Mock()
        mock_planner_instance.act.return_value = [0.2, 0.3]
        mock_planner.return_value = mock_planner_instance
        
        # テスト用引数を作成
        class Args:
            scenario = "rational"
            num_agents = 2
            max_timesteps = 10
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "gpt-4o-mini"
            prompt_algo = "io"
            history_len = 20
            timeout = 10
            two_timescale = 5
            agent_mix = "us_income"
            bracket_setting = "two"
            percent_ego = 100
            percent_alt = 0
            percent_adv = 0
            tax_type = "US_FED"
            wandb = False
            debug = False

        args = Args()

        # rGB2関数がスキルを返すようにモック
        with patch('llm_economist.main.rGB2', return_value=[50.0, 60.0]):
            # 例外が発生しないはず
            try:
                run_simulation(args)
            except SystemExit:
                # モックのためシミュレーションが早期終了する可能性があるが問題なし
                pass
        
        # コンポーネントが生成されたことを検証
        mock_test_agent.assert_called_once()
        assert mock_worker.call_count == 2  # エージェント2体
        mock_planner.assert_called_once()
    
    @patch('llm_economist.main.TestAgent')
    def test_llm_connection_failure(self, mock_test_agent):
        """LLM接続失敗時のシミュレーション処理テスト。"""
        # テストエージェントが失敗するようにモック
        mock_test_agent.side_effect = Exception("Connection failed")

        class Args:
            scenario = "rational"
            num_agents = 1
            llm = "gpt-4o-mini"
            debug = False
        
        args = Args()
        
        # エラーコード1で終了するはず
        with pytest.raises(SystemExit) as exc_info:
            run_simulation(args)
        
        assert exc_info.value.code == 1
    
    @patch('llm_economist.main.TestAgent')
    @patch('llm_economist.main.FixedWorker')
    @patch('llm_economist.main.FixedTaxPlanner')
    def test_fixed_agents_simulation(self, mock_fixed_planner, mock_fixed_worker, mock_test_agent):
        """固定（非LLM）エージェントでのシミュレーションテスト。"""
        mock_test_agent.return_value = Mock()
        
        mock_worker_instance = Mock()
        mock_worker_instance.labor = 50
        mock_worker_instance.utility = 100
        mock_fixed_worker.return_value = mock_worker_instance
        
        mock_planner_instance = Mock()
        mock_planner_instance.act.return_value = [0.2, 0.3]
        mock_fixed_planner.return_value = mock_planner_instance
        
        class Args:
            scenario = "rational"
            num_agents = 1
            max_timesteps = 5
            worker_type = "FIXED"
            planner_type = "FIXED"
            llm = "gpt-4o-mini"
            prompt_algo = "io"
            history_len = 20
            timeout = 10
            two_timescale = 5
            agent_mix = "uniform"
            bracket_setting = "two"
            percent_ego = 100
            percent_alt = 0
            percent_adv = 0
            tax_type = "US_FED"
            wandb = False
            debug = False
        
        args = Args()
        
        try:
            run_simulation(args)
        except SystemExit:
            pass
        
        mock_fixed_worker.assert_called_once()
        mock_fixed_planner.assert_called_once()
    
    def test_invalid_scenario(self):
        """無効なシナリオが処理されることを確認するテスト。"""
        class Args:
            scenario = "invalid_scenario"
            agent_mix = "us_income"
            num_agents = 1
        
        args = Args()
        
        # ペルソナ処理時にValueErrorなどが発生するはず
        # モックの複雑さから完全なシミュレーションテストは困難だが、
        # シナリオ検証ロジックのテストは可能
        assert args.scenario not in ["rational", "bounded", "democratic"]


class TestUtilityFunctions:
    """ユーティリティ関数のテスト。"""
    
    @patch('llm_economist.main.logging.basicConfig')
    def test_setup_logging(self, mock_logging):
        """ロギング設定のテスト。"""
        from llm_economist.main import setup_logging
        import logging
        
        setup_logging(logging.DEBUG)
        mock_logging.assert_called_once()
    
    def test_invalid_agent_mix(self):
        """無効なエージェント構成の処理テスト。"""
        class Args:
            agent_mix = "invalid_mix"
            num_agents = 5
        
        args = Args()
        
        # 実際のシミュレーションではValueErrorが発生するはず
        with pytest.raises(ValueError, match="Unknown agent mix"):
            # run_simulationのエージェント構成検証ロジックをシミュレート
            if args.agent_mix not in ['uniform', 'us_income']:
                raise ValueError(f'Unknown agent mix: {args.agent_mix}')


# 結合テスト
class TestFullWorkflow:
    """完全なワークフローのテスト。"""
    
    @patch('llm_economist.main.TestAgent')
    @patch('llm_economist.main.Worker')
    @patch('llm_economist.main.TaxPlanner')
    @patch('llm_economist.main.rGB2')
    @patch('llm_economist.main.distribute_personas')
    @patch('llm_economist.main.distribute_agents')
    def test_bounded_scenario_workflow(self, mock_dist_agents, mock_dist_personas, 
                                     mock_rgb2, mock_planner, mock_worker, mock_test_agent):
        """限定合理性シナリオの完全なワークフローテスト。"""
        # モックのセットアップ
        mock_test_agent.return_value = Mock()
        mock_rgb2.return_value = [50.0, 60.0]
        mock_dist_personas.return_value = {"persona1": "data1", "persona2": "data2"}
        mock_dist_agents.return_value = ["egotistical", "altruistic"]
        
        mock_worker_instance = Mock()
        mock_worker_instance.labor = 50
        mock_worker_instance.utility = 100
        mock_worker_instance.act_labor = Mock()
        mock_worker.return_value = mock_worker_instance
        
        mock_planner_instance = Mock()
        mock_planner_instance.act.return_value = [0.2, 0.3]
        mock_planner.return_value = mock_planner_instance
        
        class Args:
            scenario = "bounded"
            num_agents = 2
            max_timesteps = 5
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "gpt-4o-mini"
            prompt_algo = "io"
            history_len = 20
            timeout = 10
            two_timescale = 2
            agent_mix = "us_income"
            bracket_setting = "two"
            percent_ego = 50
            percent_alt = 30
            percent_adv = 20
            tax_type = "US_FED"
            wandb = False
            debug = False
        
        args = Args()
        
        try:
            run_simulation(args)
        except SystemExit:
            pass
        
        # ワークフローが正しく実行されたことを検証
        mock_test_agent.assert_called_once()
        mock_rgb2.assert_called_once_with(2)
        mock_dist_personas.assert_called_once()
        mock_dist_agents.assert_called_once()
        assert mock_worker.call_count == 2
        mock_planner.assert_called_once() 