import logging
from time import sleep
from ..utils.common import Message
import json
from ..models.openai_model import OpenAIModel
from ..utils.bracket import get_num_brackets, get_default_rates
from collections import Counter
import numpy as np

class LLMAgent:
    def __init__(self, llm_type: str,
                 name: str,
                 prompt_algo: str='io',
                 history_len: int=10,
                 timeout: int=10,
                 K: int=3,
                 args=None) -> None:
        assert args is not None

        self.bracket_setting = args.bracket_setting
        self.num_brackets = get_num_brackets(self.bracket_setting)
        self.tax_rates = get_default_rates(self.bracket_setting)

        self.logger = logging.getLogger('main')
        self.name = name

        # llm_typeに基づいて適切なモデルを初期化
        self.llm = self._create_llm_model(llm_type)
        
        self.history_len = history_len
        self.timeout = timeout  # 失敗する前のメッセージリトライ回数
        self.system_prompt = None   # サブクラスで上書きが必要
        self.init_message_history()

        self.prompt_algo = prompt_algo
        self.K = K  # プロンプトツリーの深さ

    def _create_llm_model(self, llm_type: str):
        """タイプに基づいて適切なLLMモデルを作成する。"""
        if llm_type == 'None':
            return None
        return OpenAIModel(model_name=llm_type)

    def act(self) -> str:
        raise NotImplementedError

    def init_message_history(self) -> None:
        # [{timestep: i, 'system_prompt': '', 'user_prompt': 'Historical timesteps: ', 'action': '' }, ...]
        # 最初のタイムステップを初期化
        self.message_history = [{
            'timestep': 0,
            'system_prompt': '',
            'user_prompt': '',
            'historical': '',
            'action': '',
            'leader': 'planner',
            'metric': 0
        }]
        return

    def add_message_history_timestep(self, timestep: int) -> None:
        assert self.system_prompt is not None
        new_msg_dict = {
            'timestep': timestep,
            'system_prompt': self.system_prompt,
            'user_prompt': '',
            'historical': '',
            'action': '',
            'leader': '',
            'metric': 0
        }
        self.message_history.append(new_msg_dict)
        return
    
    def get_historical_message(self, timestep: int, retry: bool=False, include_user_prompt: bool=True) -> str:
        unique_metrics = set()  # ユニークな'metric'値を格納するセット
        sorted_message_history = []  # ソートされたユニークエントリを格納するリスト

        # 'metric'キーで降順にソート
        for item in sorted(self.message_history, key=lambda x: x['metric'], reverse=True):
            if str(item['metric']) + str(item['action']) not in unique_metrics:
                unique_metrics.add(str(item['metric']) + str(item['action']))
                sorted_message_history.append(item)
        output = 'Historical data:\n'
        for t in range(max(0, timestep-min(self.history_len, len(self.message_history))), timestep+1):
            output += f'Timestep {t}:\n'
            output += self.message_history[t]['historical']
        N = min(5, len(sorted_message_history))
        output += f'Best {N} timesteps:\n'
        for i in range(N):
            output += f"Timestep {sorted_message_history[i]['timestep']} (leader {self.message_history[t]['leader']}):\n"
            output += sorted_message_history[i]['historical']
        if include_user_prompt:
            output += self.message_history[timestep]['user_prompt']
        if retry:
            output += "Please enter a valid response. "
        return output
    
    def act_llm(self, timestep: int, keys: list[str], parse_func, depth: int=0, retry: bool=False) -> list[float]:
        # 前のタイムステップのユーザープロンプトを連結して現在のタイムステップの履歴情報を取得
        msg = self.get_historical_message(timestep, retry)
        if self.prompt_algo == 'io':
            return self.prompt_io(msg, timestep, keys, parse_func)
        elif self.prompt_algo == 'cot':
            return self.prompt_cot(msg, timestep, keys, parse_func)
        elif self.prompt_algo == 'sc':
            return self.prompt_sc(msg, timestep, keys, parse_func)
        elif self.prompt_algo == 'tot':
            return self.prompt_sc(msg, timestep, keys, parse_func)
        elif self.prompt_algo == 'mcts':
            return self.prompt_mcts(msg, timestep, keys, parse_func)
        else:
            raise ValueError()

    
    def call_llm(self, msg: str, timestep: int, keys: list[str], parse_func, depth: int=0, retry: bool=False, cot: bool=False, temperature: float=0.7) -> list[float]:
        response_found = False
        if cot:
            llm_output, response_found = self.llm.send_msg(self.system_prompt, msg, temperature=temperature, json_format=True)
            msg = msg + llm_output
        if not response_found:
            llm_output, _ = self.llm.send_msg(self.system_prompt, msg + '\n{"', temperature=temperature, json_format=True)
        try:
            self.logger.info(f"LLM OUTPUT RECURSE {depth}\t{llm_output.strip()}")
            # JSONの中括弧 {} をパース
            data = json.loads(llm_output)
            parsed_keys = []
            for key in keys:
                parsed_keys.append(data[key])
            output = parse_func(parsed_keys)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            if depth <= self.timeout:
                return self.call_llm(msg, timestep, keys, parse_func, depth=depth+1, retry=True)
            else:
                raise ValueError(f"最大再帰深度={depth}に到達しました。JSONパースエラー: " + str(e))
        return output
    
    # プロンプティング
    def prompt_io(self, msg: str, timestep: int, keys: list[str], parse_func) -> list[float]:
        return self.call_llm(msg, timestep, keys, parse_func)
    
    # Self-Consistencyプロンプティング
    def prompt_sc(self, msg: str, timestep: int, keys: list[str], parse_func) -> list[float]:
        llm_outputs = []
        for i in range(self.K):
            llm_output = self.prompt_cot(msg, timestep, keys, parse_func)
            llm_outputs.append(llm_output)
        def most_common(lst):
            lst_str = [str(x) for x in lst]
            data = Counter(lst_str)
            str_common = data.most_common(1)[0][0]
            str_index = lst_str.index(str_common)
            return lst[str_index]
        output = most_common(llm_outputs)
        return output
    
    # Chain of Thoughtプロンプティング
    def prompt_cot(self, msg: str, timestep: int, keys: list[str], parse_func) -> list[float]:
        cot_prompt = " Let's think step by step. Your thought should no more than 4 sentences."
        # エージェントのuser_promptにJSON形式の思考 "thought":"<step-by-step-thinking>" レスポンスを常に追加
        return self.call_llm(msg + cot_prompt, timestep, keys, parse_func, cot=True)
    
    def add_message(self, timestep: int, m_type: Message, **args) -> None:
        raise NotImplementedError
    
    def parse_tax(self, items: list[str]) -> tuple:
        # self.logger.info("[parse_tax]", tax_rates)
        tax_rates = items[0]
        output_tax_rates = []
        if len(tax_rates) != self.num_brackets:  # 税率区分数の固定チェック
            raise ValueError('税率値が多すぎます', tax_rates)
        for i, rate in enumerate(tax_rates):
            if isinstance(rate, str):
                rate = rate.replace('$','').replace(',','').replace('%', '')
            rate = float(rate)
            rate = np.clip(rate, -self.delta, self.delta)
            rate = np.round(rate / 10) * 10
            # rate = np.round(rate / 10) * 10
            if rate + self.tax_rates[i] > 100:
                rate = 100 - self.tax_rates[i]
            elif rate + self.tax_rates[i] < 0:
                rate = - self.tax_rates[i]
            # if rate > 100: rate = 100
            # if rate > 100 or rate < 0:
            #     raise ValueError(f'Rates outside bounds: 0 <= {rate} <= 100')
            output_tax_rates.append(rate)
        # return (output_tax_rates, float(items[1]))
        return (output_tax_rates,)
    
class TestAgent(LLMAgent):
    def __init__(self, llm: str, args):
        super().__init__(llm, name='TestAgent', args=args)
        max_retries = 5  # 最大試行回数 (初回を含む)
        initial_delay = 1  # 初期遅延 (秒)
        max_delay = 60  # リトライ間の最大遅延
        current_delay = initial_delay

        for attempt in range(max_retries):
            try:
                self.llm.send_msg('', 'This is a test. Output \"test\" in response.')
                print("OpenAI LLMサービスへの接続に成功しました")
                return  # 成功時に終了
            except Exception as e:
                if attempt == max_retries - 1:  # 最後の試行が失敗
                    raise RuntimeError(
                        f"{max_retries}回の試行後に接続に失敗しました。最後のエラー: {str(e)}"
                    ) from e

                print(f"試行 {attempt + 1} 失敗。{current_delay}秒後にリトライ...")
                sleep(current_delay)
                current_delay = min(current_delay * 2, max_delay)  # 上限付き指数バックオフ
