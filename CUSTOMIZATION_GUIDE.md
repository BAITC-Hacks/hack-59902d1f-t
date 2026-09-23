# Практическое руководство по кастомизации агента

## Как адаптировать решение под разные сценарии

---

## 1. Изменение баланса разведка vs эксплуатация

### Сценарий: "Консервативный подход" (мало бюджета на риск)

**Текущие параметры** (в `agent.py`):
```python
pilots_budget = int(env.remaining_budget * 0.4)  # 40% на разведку
max_pilots = min(15, env.pilots_left)
```

**Измените на**:
```python
pilots_budget = int(env.remaining_budget * 0.25)  # 25% на разведку
max_pilots = min(8, env.pilots_left)  # меньше пилотов

# Также увеличьте консервативность при масштабировании:
uncertainty_discount = 0.70  # было 0.8
```

**Эффект**: Больше бюджета на финальные кампании, но с менее проверенными стратегиями

---

### Сценарий: "Агрессивная разведка" (есть время на пилоты)

**Измените на**:
```python
pilots_budget = int(env.remaining_budget * 0.50)  # 50% на разведку
max_pilots = min(20, env.pilots_left)  # максимум пилотов

# Мение консервативно при масштабировании:
uncertainty_discount = 0.90  # было 0.8
```

**Эффект**: Больше информации о сегментах, но риск переизучения

---

## 2. Настройка Thompson Sampling

### Изменение prior'а

**Текущий prior** (в `BayesianContextualBandit`):
```python
def sample_posterior(self, key):
    alpha = arm['alpha'] + n_success
    beta = arm['beta'] + (100 - n_trials)  # оптимистичный!
```

**Оптимистичный prior** (более смелый):
```python
beta = arm['beta'] + max(0, 50 - n_trials)  # более оптимистичный
```
→ Раньше выбирает неизученные arms

**Пессимистичный prior** (более осторожный):
```python
beta = arm['beta'] + (150 - n_trials)  # более пессимистичный
```
→ Больше доверяет пилотам, меньше пробует новое

---

## 3. Адаптивный размер пилотов

### Текущая логика:
```python
n_pilot = min(100, max(20, int(np.sqrt(len(env.customer_profile) / 100))))
# Результат: ~48 клиентов на пилот (для 23K total)
```

### Вариант 1: Зависимо от uncertainty

```python
mean, std = self.bandit.get_expected_value(key)

if std == float('inf'):
    n_pilot = 150  # неизведано → большой пилот
elif std > 5000:
    n_pilot = 100  # высокая uncertainty
elif std > 1000:
    n_pilot = 50   # средняя
else:
    n_pilot = 20   # низкая uncertainty → малый пилот
```

### Вариант 2: Зависимо от бюджета

```python
remaining_budget = env.remaining_budget - sum([p['cost'] for p in self.pilot_results])
remaining_pilots = env.pilots_left - len(self.pilot_results)

avg_pilot_budget = remaining_budget * 0.4 / max(remaining_pilots, 1)
channel_cost = {'push': 0, 'sms': 4, 'digital_ads': 22, 'call': 160}[channel]

n_pilot = int(avg_pilot_budget / max(channel_cost, 1))
n_pilot = min(200, max(10, n_pilot))  # clamp
```

---

## 4. Выбор сегментов для пилотов

### Текущий подход: Thompson sampling всех комбинаций

### Альтернатива 1: Только HIGH ARPU сегменты

```python
def _extract_segments(self):
    segments = []
    # Только HIGH ARPU
    for data in ['LITE', 'HEAVY']:  # skip NON_USER
        for call in ['MEDIUM', 'HIGH']:  # skip LOW
            segments.append(f"A:HIGH|D:{data}|C:{call}")
    return segments
    # ~12 сегментов вместо 27
```

**Эффект**: Быстрее находим TOP сегменты, но может пропустить "hidden gems" в LOW ARPU

### Альтернатива 2: Иерархический выбор

```python
def select_arms_hierarchical(self):
    # Шаг 1: Выбираем лучший ARPU сегмент
    # Шаг 2: Для каждого ARPU, выбираем лучший Data segment
    # Шаг 3: Для каждого (ARPU, Data), выбираем лучший Call segment
    # Итого: 3 + 3 + 3 = 9 пилотов вместо 15
```

---

## 5. Выбор каналов

### Текущая логика: Все 4 канала (push, sms, digital_ads, call)

### Оптимизация 1: Фильтр по channel effectiveness

```python
def _get_effective_channels(self, segment_key):
    arpu = segment_key.split('|')[0]
    
    if 'HIGH' in arpu:
        return ['call', 'digital_ads']  # дорогие каналы для VIP
    elif 'MID' in arpu:
        return ['sms', 'digital_ads']   # средние каналы
    else:
        return ['push', 'sms']           # дешёвые каналы
```

**Эффект**: Сокращает комбинаций в 2-3 раза, фокусируется на релевантных

### Оптимизация 2: Динамический выбор на основе бюджета

```python
remaining = env.remaining_budget / env.pilots_left

if remaining < 50:
    channels = ['push', 'sms']  # дешёвые каналы
elif remaining < 200:
    channels = ['push', 'sms', 'digital_ads']
else:
    channels = ['push', 'sms', 'digital_ads', 'call']  # всё
```

---

## 6. Оптимизация портфеля

### Текущий подход: Integer Programming

### Альтернатива 1: Увеличить макс. кампаний

```python
# вместо:
optimizer = IntegerPortfolioOptimizer(top_campaigns, env.remaining_budget, max_campaigns=10)

# на:
optimizer = IntegerPortfolioOptimizer(top_campaigns, env.remaining_budget, max_campaigns=15)
```

**Риск**: Можем выбрать слабые кампании, но больше охват

### Альтернатива 2: Добавить constraint на diversity

```python
# Убедиться что выбираем разные сегменты (не все на HIGH ARPU)
# In PuLP:
prob += pulp.lpSum([x[i] for i if campaigns[i]['filter_arpu_segment'] == 'HIGH']) <= 5
prob += pulp.lpSum([x[i] for i if campaigns[i]['channel'] == 'call']) <= 3
```

**Эффект**: Более сбалансированный портфель, меньше "all eggs in one basket"

---

## 7. Обработка пилотов

### Текущая логика: Берём среднее от пилота

### Оптимизация 1: Взвешенное среднее по размеру

```python
def update_arm_weighted(self, key, pilot_results):
    # Берём среднее, взвешивая по n_customers
    total_gain = sum([p['gain'] * p['n_customers'] 
                      for p in pilot_results if p['arm'] == key])
    total_customers = sum([p['n_customers'] 
                          for p in pilot_results if p['arm'] == key])
    
    avg_gain = total_gain / max(total_customers, 1)
```

### Оптимизация 2: Exponential smoothing

```python
def update_arm_ema(self, key, new_observation):
    arm = self.arms[key]
    
    # Exponential moving average
    alpha = 0.7  # weight новых пилотов
    arm['ema_gain'] = (alpha * new_observation + 
                       (1-alpha) * arm.get('ema_gain', 0))
```

**Эффект**: Больше внимания последним (свежим) результатам

---

## 8. Использование LLM (agent_advanced.py)

### Включение/выключение LLM

```python
# Отключить (если OPENAI_API_KEY не установлен):
self.llm = None  # или лучше: use fallback
```

### Кастомные промпты

```python
def analyze_pilot_results(self, pilot_results):
    summary = self._format_results(pilot_results)
    
    # Измените промпт:
    prompt = f"""
    Телеком оператор. Пилотные кампании по смене тарифов.
    
    Фокус анализа:
    1. Какие каналы (push/sms/call/digital) наиболее эффективны?
    2. Какие тарифы клиенты предпочитают?
    3. Есть ли неожиданные тренды?
    
    Результаты:
    {summary}
    
    Выводы (2-3 предложения):
    """
    
    return self.client.chat.completions.create(...)
```

---

## 9. Хранение состояния (для multi-round experiments)

### Сохранение posteriori между раундами

```python
def save_bandit_state(self, filename='bandit_state.json'):
    state = {}
    for key, arm in self.bandit.arms.items():
        state[str(key)] = {
            'alpha': arm['alpha'],
            'beta': arm['beta'],
            'n_trials': arm['n_trials'],
            'observed_gains': arm['observed_gains']
        }
    
    with open(filename, 'w') as f:
        json.dump(state, f)

def load_bandit_state(self, filename='bandit_state.json'):
    with open(filename, 'r') as f:
        state = json.load(f)
    
    for key_str, data in state.items():
        key = eval(key_str)  # convert back to tuple
        self.bandit.arms[key].update(data)
```

---

## 10. Пример: полная кастомизация

```python
# agent_custom.py

from agent import Agent, BayesianContextualBandit, IntegerPortfolioOptimizer

class CustomAgent(Agent):
    """Агент с кастомными параметрами"""
    
    def _exploration_phase(self, env):
        # 30% на разведку (вместо 40%)
        pilots_budget = int(env.remaining_budget * 0.30)
        
        # Только топ-5 ARPU сегментов + важные каналы
        candidates = self.bandit.select_arms_thompson(n_candidates=10)
        
        # Адаптивный размер пилота на основе remaining budget
        for i, (segment_key, target_tariff, channel) in enumerate(candidates):
            remaining_budget = env.remaining_budget - sum([...])
            remaining_pilots = env.pilots_left - i
            
            n_pilot = int((remaining_budget * 0.3) / 
                         (remaining_pilots * channel_cost[channel]))
            n_pilot = min(150, max(30, n_pilot))
            
            # Run pilot
            ...
    
    def _optimize_portfolio(self, campaigns, env):
        # Добавляем diversity constraint
        if HAS_PULP:
            return self._optimize_with_diversity(campaigns, env)
        else:
            return super()._optimize_portfolio(campaigns, env)
    
    def _optimize_with_diversity(self, campaigns, env):
        # Гарантируем мин 2 разных канала, мин 2 разных тарифа
        ...

# Использование:
if __name__ == "__main__":
    agent = CustomAgent()
    # agent.act(env) будет использовать кастомные параметры
```

---

## Чеклист оптимизации

- [ ] Отрегулировать баланс разведка/эксплуатация (40% по умолчанию)
- [ ] Выбрать prior для Thompson sampling (оптимистичный/пессимистичный)
- [ ] Установить размер пилотов (адаптивный vs фиксированный)
- [ ] Выбрать сегменты для пилотов (все 27 vs избранные)
- [ ] Выбрать каналы (фильтровать по ARPU или бюджету)
- [ ] Настроить Integer Programming (макс. кампаний, diversity)
- [ ] Включить/выключить LLM (если API доступен)
- [ ] Тестировать на local_eval.py --runs 10 (проверить устойчивость)
- [ ] Анализировать пилот_results.csv для insights

---

## Полезные метрики для мониторинга

```python
# В конце act():
print(f"[Metrics]")
print(f"  Пилотов: {len(self.pilot_results)}")
print(f"  Avg pilot gain: {np.mean([p['observed_gain'] for p in self.pilot_results]):.0f}")
print(f"  Avg ROI: {np.mean([p['roi'] for p in self.pilot_results]):.2f}")
print(f"  Max uncertainty: {max([p.get('uncertainty', 0) for p in self.pilot_results]):.0f}")
print(f"  Campaigns selected: {len(result)}")
print(f"  Est total gain: {sum([c['expected_gain'] for c in result]):.0f}")
```

Сохранить в CSV для анализа между раундами:
```python
pd.DataFrame(self.pilot_results).to_csv('pilot_results.csv', index=False)
```
