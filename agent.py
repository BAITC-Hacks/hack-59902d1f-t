"""
Продвинутая версия агента с:
- Contextual Thompson sampling (features-driven)
- LLM для интерпретации результатов пилотов
- Адаптивный размер пилотов на основе uncertainty
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import pulp
    HAS_PULP = True
except:
    HAS_PULP = False


class ContextualBayesianBandit:
    """Contextual bandit с features-driven оценкой"""
    
    def __init__(self, customer_profile: pd.DataFrame, tariffs: list, channels: list):
        self.customer_profile = customer_profile
        
        if isinstance(tariffs, pd.DataFrame) and 'tariff_plan_code' in tariffs.columns:
            self.tariffs = tariffs['tariff_plan_code'].tolist()
        else:
            self.tariffs = list(tariffs)
            
        self.channels = channels
        
        self.segments = self._extract_segments()
        self.arms = {}
        self._initialize_arms()
    
    def _extract_segments(self) -> List[str]:
        """Генерируем сегменты с контекстными features"""
        segments = []
        for arpu in ['LOW', 'MID', 'HIGH']:
            for data in ['NON_USER', 'LITE', 'HEAVY']:
                for call in ['LOW', 'MEDIUM', 'HIGH']:
                    segments.append(f"A:{arpu}|D:{data}|C:{call}")
        return segments
    
    def _initialize_arms(self):
        """Инициализируем вооружение с Beta(1, 1) prior"""
        for segment in self.segments:
            for tariff in self.tariffs:
                for channel in self.channels:
                    key = (segment, tariff, channel)
                    self.arms[key] = {
                        'segment': segment,
                        'tariff': tariff,
                        'channel': channel,
                        'alpha': 1.0,
                        'beta': 1.0,
                        'n_trials': 0,
                        'observed_gains': [],
                        'costs': []
                    }
    
    def sample_posterior(self, key: Tuple) -> float:
        """Thompson sampling: сэмплируем из posterior"""
        arm = self.arms.get(key)
        if not arm:
            return 0.0
        
        # Beta posterior
        alpha = arm['alpha'] + sum(1 for g in arm['observed_gains'] if g > 0)
        beta = arm['beta'] + sum(1 for g in arm['observed_gains'] if g <= 0)
        
        return np.random.beta(alpha, beta)
    
    def select_arms_contextual(self, n_candidates: int = 15) -> List[Tuple]:
        """Выбираем лучшие вооружения через Thompson sampling с учетом стоимости канала"""
        channel_costs = {'push': 1, 'sms': 4, 'digital_ads': 22, 'call': 160} # 1 для push чтобы избежать / 0
        scores = []
        for key, arm in self.arms.items():
            sample = self.sample_posterior(key)
            channel = key[2]
            
            # На этапе первоначальной разведки штрафуем дорогие каналы
            if arm['n_trials'] == 0:
                cost_penalty = channel_costs.get(channel, 10) ** 0.5
                adjusted_score = sample / cost_penalty
            else:
                adjusted_score = sample
                
            scores.append((key, adjusted_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in scores[:n_candidates]]
    
    def update_arm(self, key: Tuple, n_customers: int, observed_gain: float, cost: float):
        """Обновляем posterior после пилота"""
        if key not in self.arms:
            self.arms[key] = {
                'segment': key[0],
                'tariff': key[1],
                'channel': key[2],
                'alpha': 1.0,
                'beta': 1.0,
                'n_trials': 0,
                'observed_gains': [],
                'costs': []
            }
        
        arm = self.arms[key]
        arm['n_trials'] += n_customers
        arm['observed_gains'].append(observed_gain)
        arm['costs'].append(cost)
    
    def get_expected_value(self, key: Tuple) -> Tuple[float, float]:
        """Возвращаем E[gain] и std для uncertainty quantification"""
        arm = self.arms.get(key)
        if not arm or not arm['observed_gains']:
            return 0.0, float('inf')  # максимальная неопределённость
        
        gains = np.array(arm['observed_gains'])
        mean = np.mean(gains)
        std = np.std(gains) if len(gains) > 1 else float('inf')
        
        return mean, std


class LLMAdvisor:
    """Использует LLM для интерпретации результатов пилотов"""
    
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.use_llm = self.api_key is not None
    
    def analyze_pilot_results(self, pilot_results: List[Dict]) -> str:
        """Анализирует результаты пилотов через LLM"""
        if not self.use_llm or not pilot_results:
            return ""
        
        try:
            import openai
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
            
            # Подготавливаем summary результатов
            summary = "Результаты пилотных кампаний:\n"
            for p in pilot_results[:10]:  # берём топ-10
                summary += f"- {p['segment']} -> {p['target_tariff']} ({p['channel']}): "
                summary += f"Gain={p['observed_gain']:.0f}, ROI={p['roi']:.2f}\n"
            
            response = client.chat.completions.create(
                model="openrouter/auto",
                messages=[{
                    "role": "user",
                    "content": f"""Проанализируй результаты пилотных маркетинговых кампаний 
                    для телеком оператора. Какие сегменты и каналы выглядят наиболее перспективными? 
                    Какие риски видишь в масштабировании? Будь кратким (2-3 предложения).
                    
                    {summary}"""
                }],
                max_tokens=200
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"[LLM недоступен: {str(e)[:50]}]"


class Agent:
    """Продвинутая версия агента с contextual features и LLM"""
    
    def __init__(self):
        self.bandit = None
        self.llm = LLMAdvisor()
        self.pilot_results = []
    
    def act(self, env) -> List[Dict]:
        """Главный метод агента"""
        
        print("[Agent] Инициализация contextual bandit")
        self.bandit = ContextualBayesianBandit(
            env.customer_profile,
            env.tariffs,
            env.channels
        )
        
        # Фаза 1: Умная разведка
        print("[Agent] Фаза 1: Contextual exploration")
        self._contextual_exploration(env)
        
        # Фаза 2: LLM анализ результатов
        if self.pilot_results:
            analysis = self.llm.analyze_pilot_results(self.pilot_results)
            print(f"[LLM Analysis]\n{analysis}\n")
        
        # Фаза 3: Построение кампаний
        print("[Agent] Фаза 2: Построение кампаний")
        campaigns = self._build_campaigns(env)
        
        # Фаза 4: Оптимизация портфеля
        print("[Agent] Фаза 3: Оптимизация портфеля")
        selected = self._optimize_portfolio(campaigns, env)
        
        return self._format_campaigns(selected)
    
    def _contextual_exploration(self, env):
        """Разведка с adaptive pilot sizing"""
        
        budget_for_exploration = int(env.remaining_budget * 0.35)
        max_pilots = min(18, env.pilots_left)
        
        print(f"[Exploration] Макс {max_pilots} пилотов, бюджет {budget_for_exploration}")
        
        # Выбираем кандидаты
        candidates = self.bandit.select_arms_contextual(n_candidates=max_pilots)
        
        channel_costs = {'push': 0, 'sms': 4, 'digital_ads': 22, 'call': 160}
        pilots_run = 0
        total_pilot_cost = 0
        
        for segment_key, target_tariff, channel in candidates:
            if pilots_run >= max_pilots or env.pilots_left <= 0:
                break
            
            # Адаптивный размер пилота на основе uncertainty
            mean, std = self.bandit.get_expected_value((segment_key, target_tariff, channel))
            
            if np.isinf(std):
                # Максимальная неопределённость → больший пилот
                n_pilot = 100
            else:
                # Меньше std → меньше нужен пилот
                n_pilot = min(200, max(20, int(50 * (1 + std / 10000))))
            
            cost = channel_costs.get(channel, 0) * n_pilot
            
            if total_pilot_cost + cost > budget_for_exploration:
                break
            
            try:
                filter_arpu, filter_data, filter_call = self._parse_segment(segment_key)
                
                result = env.run_pilot(
                    target_tariff=target_tariff,
                    channel=channel,
                    n_customers=n_pilot,
                    filter_arpu_segment=filter_arpu,
                    filter_data_segment=filter_data,
                    filter_call_segment=filter_call
                )
                
                if result:
                    observed_gain = result.get('total_gain', 0)
                    self.bandit.update_arm(
                        (segment_key, target_tariff, channel),
                        n_pilot,
                        observed_gain,
                        cost
                    )
                    
                    self.pilot_results.append({
                        'segment': segment_key,
                        'target_tariff': target_tariff,
                        'channel': channel,
                        'n_customers': n_pilot,
                        'observed_gain': observed_gain,
                        'cost': cost,
                        'roi': observed_gain / max(cost, 1),
                        'uncertainty': std
                    })
                    
                    pilots_run += 1
                    total_pilot_cost += cost
                    
                    print(f"  [{pilots_run}] {segment_key} -> {target_tariff} ({channel})")
                    print(f"      Gain={observed_gain:.0f}, Cost={cost}, Uncertainty={std:.0f}")
            
            except Exception as e:
                print(f"  [Error] {e}")
                continue
    
    def _parse_segment(self, segment_key: str) -> Tuple[str, str, str]:
        """Парсим ключ сегмента"""
        parts = segment_key.split('|')
        arpu = parts[0].split(':')[1] if len(parts) > 0 else 'HIGH'
        data = parts[1].split(':')[1] if len(parts) > 1 else 'HEAVY'
        call = parts[2].split(':')[1] if len(parts) > 2 else 'MEDIUM'
        return arpu, data, call
    
    def _build_campaigns(self, env) -> List[Dict]:
        """Строим кампании на основе пилотов"""
        
        campaigns = []
        
        if self.pilot_results:
            pilot_df = pd.DataFrame(self.pilot_results)
            good_pilots = pilot_df[pilot_df['observed_gain'] > 0].sort_values('roi', ascending=False)
            
            for _, pilot in good_pilots.iterrows():
                segment_key = pilot['segment']
                target_tariff = pilot['target_tariff']
                channel = pilot['channel']
                observed_gain = pilot['observed_gain']
                n_pilot = pilot['n_customers']
                
                filter_arpu, filter_data, filter_call = self._parse_segment(segment_key)
                
                # Фильтруем клиентов
                filtered = env.customer_profile
                if filter_arpu:
                    filtered = filtered[filtered['arpu_segment'] == filter_arpu]
                if filter_data:
                    filtered = filtered[filtered['data_segment'] == filter_data]
                if filter_call:
                    filtered = filtered[filtered['call_segment'] == filter_call]
                
                segment_size = len(filtered)
                if segment_size == 0:
                    continue
                
                # Gain per customer с uncertainty-aware скидкой
                gain_per_customer = observed_gain / max(n_pilot, 1)
                
                # Консервативнее для высокой uncertainty
                uncertainty_discount = 0.7 if pilot['uncertainty'] > 5000 else 0.85
                
                expected_gain_full = gain_per_customer * segment_size * uncertainty_discount
                
                channel_costs = {'push': 0, 'sms': 4, 'digital_ads': 22, 'call': 160}
                cost_per_contact = channel_costs.get(channel, 0)
                
                max_customers = min(segment_size, 5000)
                contact_cost = cost_per_contact * max_customers
                
                expected_net_gain = expected_gain_full * (max_customers / segment_size) - contact_cost
                
                campaigns.append({
                    'segment_key': segment_key,
                    'filter_arpu_segment': filter_arpu,
                    'filter_data_segment': filter_data,
                    'filter_call_segment': filter_call,
                    'target_tariff': target_tariff,
                    'channel': channel,
                    'n_customers': max_customers,
                    'cost': contact_cost,
                    'expected_gain': expected_gain_full * (max_customers / segment_size),
                    'expected_net_gain': expected_net_gain
                })
        
        return campaigns
    
    def _optimize_portfolio(self, campaigns: List[Dict], env) -> List[Dict]:
        """Оптимизируем выбор кампаний"""
        
        if not campaigns:
            return []
        
        viable = [c for c in campaigns if c['cost'] <= env.remaining_budget and c['n_customers'] <= env.remaining_contacts]
        if not viable:
            return []
        
        viable.sort(key=lambda x: x['expected_net_gain'], reverse=True)
        top_campaigns = viable[:15]
        
        # Integer programming
        if HAS_PULP and len(top_campaigns) > 3:
            return self._optimize_with_pulp(top_campaigns, env.remaining_budget, env.remaining_contacts)
        else:
            return self._optimize_greedy(top_campaigns, env.remaining_budget, env.remaining_contacts)
    
    def _optimize_with_pulp(self, campaigns: List[Dict], budget: float, contacts_limit: int) -> List[Dict]:
        """Integer programming optimization"""
        n = len(campaigns)
        prob = pulp.LpProblem("CampaignSelection", pulp.LpMaximize)
        
        x = [pulp.LpVariable(f"campaign_{i}", cat='Binary') for i in range(n)]
        
        gains = [c['expected_gain'] for c in campaigns]
        costs = [c['cost'] for c in campaigns]
        contacts = [c['n_customers'] for c in campaigns]
        
        prob += pulp.lpSum([gains[i] * x[i] for i in range(n)])
        prob += pulp.lpSum([costs[i] * x[i] for i in range(n)]) <= budget
        prob += pulp.lpSum([contacts[i] * x[i] for i in range(n)]) <= contacts_limit
        prob += pulp.lpSum(x) <= 10
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        selected = [campaigns[i] for i in range(n) if x[i].varValue > 0.5]
        return selected
    
    def _optimize_greedy(self, campaigns: List[Dict], budget: float, contacts_limit: int) -> List[Dict]:
        """Жадный алгоритм"""
        selected = []
        spent_budget = 0
        spent_contacts = 0
        
        for campaign in campaigns:
            if len(selected) >= 10:
                break
            cost = campaign['cost']
            contacts = campaign['n_customers']
            if spent_budget + cost <= budget and spent_contacts + contacts <= contacts_limit:
                selected.append(campaign)
                spent_budget += cost
                spent_contacts += contacts
        
        return selected
    
    def _format_campaigns(self, campaigns: List[Dict]) -> List[Dict]:
        """Форматируем результат"""
        result = []
        for i, c in enumerate(campaigns):
            formatted = {
                'campaign_name': f"Campaign_{i+1}_{c['target_tariff']}",
                'target_tariff': c['target_tariff'],
                'channel': c['channel']
            }
            
            if c.get('filter_arpu_segment'):
                formatted['filter_arpu_segment'] = c['filter_arpu_segment']
            if c.get('filter_data_segment'):
                formatted['filter_data_segment'] = c['filter_data_segment']
            if c.get('filter_call_segment'):
                formatted['filter_call_segment'] = c['filter_call_segment']
            
            result.append(formatted)
        
        return result


if __name__ == "__main__":
    agent = Agent()
