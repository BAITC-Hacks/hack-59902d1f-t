"""
agent.py — MVP-агент для Beeline Tariff Marketing Campaigns Case.

Стратегия одним абзацем:
  1. Строим СВОЙ априор эффекта по ячейкам (current_tariff, arpu_segment) ->
     target_tariff из выданной истории data/change_tariff.csv (то же знание,
     что у любого участника с самого начала).
  2. Разведываем самые перспективные по априору ячейки ПИЛОТАМИ на канале
     push — он бесплатный (cost_per_contact = 0), поэтому пилотировать им
     можно по максимуму (до 200 абонентов), не трогая денежный бюджет.
     Две волны: сначала широко и дёшево (много ячеек, пилот поменьше),
     затем — дозаход крупным пилотом по лидерам первой волны, чтобы снизить
     шум именно там, где ошибка дороже всего.
  3. Комбинируем априор и наблюдения через обратно-взвешенное среднее
     (весом идёт число наблюдений — и историческое, и пилотное), получаем
     апостериорную оценку "сырого" эффекта raw = arpu_change_pct *
     conversion_rate, который не зависит от канала.
  4. Для каждой ячейки раз считаем ожидаемую чистую выгоду на контакт для
     ВСЕХ каналов (raw * channel_multiplier * avg_arpu − cost_per_contact) и
     берём канал с максимумом — часто это не всегда push, потому что дорогой
     звонок может окупаться на дорогих абонентах.
  5. Берём с консервативной поправкой (mean − k·std) до 10 непересекающихся
     по ячейкам кампаний с наибольшей ожидаемой суммарной выгодой, пока
     хватает бюджета/охвата.

Ячейки в финальных кампаниях не пересекаются (каждая ячейка current_tariff x
arpu_segment используется максимум в одной кампании), поэтому дедупликация
абонентов между кампаниями в scoring_core их не портит.
"""

import os

import numpy as np
import pandas as pd

ARPU_BINS = [-np.inf, 1000, 5000, np.inf]
ARPU_LABELS = ["LOW", "MID", "HIGH"]

PER_CUSTOMER_STD = 0.804  # заявленный в среде разброс эффекта на одного абонента

MAX_CAMPAIGNS = 10
MAX_CUSTOMERS_PER_CAMPAIGN = 5000
PILOT_CHANNEL = "push"  # бесплатная разведка

STAGE1_N = 60          # размер пилота первой (широкой) волны
STAGE2_N = 200          # размер пилота второй (уточняющей) волны
STAGE1_COUNT = 12       # сколько ячеек разведываем в первой волне
STAGE2_COUNT = 6        # сколько лидеров дозаходим второй волной
UNCERTAINTY_K = 0.6      # консервативная поправка: используем mean - k*std


def _find_change_tariff_path():
    for p in ("data/change_tariff.csv", "change_tariff.csv",
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "change_tariff.csv")):
        if os.path.exists(p):
            return p
    return None


def _build_prior(change_tariff: pd.DataFrame) -> pd.DataFrame:
    """Априорная оценка raw_effect = arpu_change_pct * conversion_rate по истории.

    conversion_rate здесь — доля переходов именно в этот target среди всех
    исторических переходов из ячейки (прокси вероятности конверсии, а не сама
    вероятность — но для ранжирования кандидатов этого достаточно).
    """
    df = change_tariff.copy()
    df["arpu_segment"] = pd.cut(df["AVG_ARPU_PREV_3M"], bins=ARPU_BINS, labels=ARPU_LABELS)
    df = df[df["AVG_ARPU_PREV_3M"] >= 100].copy()
    df["arpu_change_pct"] = ((df["AVG_ARPU_NEXT_3M"] - df["AVG_ARPU_PREV_3M"])
                              / df["AVG_ARPU_PREV_3M"]).clip(-1, 3)

    grouped = (df.groupby(["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"], observed=True)
               .agg(arpu_change_pct=("arpu_change_pct", "mean"), count=("ID_NUMBER", "size"))
               .reset_index())
    totals = (grouped.groupby(["tariff_plan_code_from", "arpu_segment"], observed=True)["count"]
              .sum().rename("total").reset_index())
    grouped = grouped.merge(totals, on=["tariff_plan_code_from", "arpu_segment"])
    grouped["conversion_rate"] = grouped["count"] / grouped["total"]
    grouped["raw_effect"] = grouped["arpu_change_pct"] * grouped["conversion_rate"]
    grouped["prior_n"] = grouped["count"]
    return grouped.rename(columns={"tariff_plan_code_from": "current_tariff",
                                    "tariff_plan_code_to": "target_tariff"})[
        ["current_tariff", "target_tariff", "arpu_segment", "raw_effect", "prior_n"]]


def _price_fallback_raw(current_tariff, target_tariff, dict_tariff):
    """Для пар без истории: слабый консервативный прокси по разнице цен тарифов."""
    price = dict_tariff.set_index("tariff_plan_code")["price_tariff"]
    if current_tariff not in price.index or target_tariff not in price.index:
        return 0.0
    scale = max(price.median(), 1.0)
    ratio = (price[target_tariff] - price[current_tariff]) / scale
    return float(np.clip(ratio * 0.4, -1.0, 3.0)) * 0.15


def _best_channel(raw_effect, avg_arpu, channels):
    """Канал с максимальной ожидаемой чистой выгодой на контакт для данной ячейки."""
    best_channel, best_net = None, -np.inf
    for ch, spec in channels.items():
        ratio = float(np.clip(raw_effect * spec["conversion_multiplier"], -1.0, 2.0))
        net_per_contact = ratio * avg_arpu - spec["cost_per_contact"]
        if net_per_contact > best_net:
            best_channel, best_net = ch, net_per_contact
    return best_channel, best_net


class Agent:
    def __init__(self, change_tariff=None):
        self.change_tariff = change_tariff

    def act(self, env):
        profile = env.customer_profile
        channels = env.channels
        dict_tariff = env.tariffs
        known_tariffs = list(dict_tariff["tariff_plan_code"])
        push_mult = channels[PILOT_CHANNEL]["conversion_multiplier"]

        cells = (profile.groupby(["current_tariff", "arpu_segment"], observed=True)
                 .agg(n=("ID_NUMBER", "size"), avg_arpu=("predicted_arpu", "mean"))
                 .reset_index())

        # --- 1. Априор из истории ---
        prior = pd.DataFrame()
        ct_path = _find_change_tariff_path() if self.change_tariff is None else None
        if self.change_tariff is not None or ct_path:
            try:
                history = self.change_tariff if self.change_tariff is not None else pd.read_csv(ct_path)
                prior = _build_prior(history)
            except Exception:
                prior = pd.DataFrame()

        # Для каждой ячейки берём ЛУЧШИЙ по априору таргет-тариф — так кандидаты
        # сразу получаются непересекающимися по ячейкам.
        candidates = []
        for _, cell in cells.iterrows():
            cell_prior = (prior[(prior.current_tariff == cell.current_tariff) &
                                 (prior.arpu_segment == cell.arpu_segment)]
                          if len(prior) else pd.DataFrame())
            best_for_cell = None
            for target in known_tariffs:
                if target == cell.current_tariff:
                    continue
                row = cell_prior[cell_prior.target_tariff == target] if len(cell_prior) else pd.DataFrame()
                if len(row):
                    raw, n_obs = float(row.raw_effect.iloc[0]), int(row.prior_n.iloc[0])
                else:
                    raw, n_obs = _price_fallback_raw(cell.current_tariff, target, dict_tariff), 0
                if best_for_cell is None or raw > best_for_cell["prior_raw"]:
                    best_for_cell = {"current_tariff": cell.current_tariff, "arpu_segment": cell.arpu_segment,
                                      "target_tariff": target, "n": int(cell.n), "avg_arpu": float(cell.avg_arpu),
                                      "prior_raw": raw, "prior_n": n_obs}
            if best_for_cell is not None:
                candidates.append(best_for_cell)

        for c in candidates:
            c["prior_value"] = max(c["prior_raw"], 0) * c["avg_arpu"] * min(c["n"], MAX_CUSTOMERS_PER_CAMPAIGN)
        candidates.sort(key=lambda c: c["prior_value"], reverse=True)

        # --- 2. Пилоты (push, бесплатно), две волны ---
        pilot_obs = {}  # (current_tariff, arpu_segment, target_tariff) -> [(ratio, n), ...]

        def run_safe(cand, n):
            if env.pilots_left <= 0:
                return None
            try:
                res = env.run_pilot(target_tariff=cand["target_tariff"], channel=PILOT_CHANNEL,
                                     n_customers=n, filter_arpu_segment=cand["arpu_segment"],
                                     filter_current_tariff=cand["current_tariff"])
            except (RuntimeError, ValueError):
                return None
            key = (cand["current_tariff"], cand["arpu_segment"], cand["target_tariff"])
            pilot_obs.setdefault(key, []).append((res["observed_lift_ratio"], res["n_customers"]))
            return res

        def posterior_raw(cand):
            """Обратно-взвешенное среднее априора и всех наблюдённых пилотов."""
            key = (cand["current_tariff"], cand["arpu_segment"], cand["target_tariff"])
            obs = pilot_obs.get(key, [])
            num = cand["prior_raw"] * max(cand["prior_n"], 1)
            den = max(cand["prior_n"], 1)
            for ratio, n in obs:
                raw = ratio / push_mult
                num += raw * n
                den += n
            mean = num / den
            std = (PER_CUSTOMER_STD / push_mult) / np.sqrt(max(den, 1))
            return mean, std

        stage1 = candidates[:STAGE1_COUNT]
        for cand in stage1:
            run_safe(cand, STAGE1_N)
        for cand in stage1:
            cand["post_raw"], cand["post_std"] = posterior_raw(cand)

        stage2 = sorted(stage1, key=lambda c: c["post_raw"] - UNCERTAINTY_K * c["post_std"], reverse=True)[:STAGE2_COUNT]
        for cand in stage2:
            run_safe(cand, STAGE2_N)

        # остаток пилотов — доразведка следующих по очереди кандидатов
        rest = candidates[STAGE1_COUNT:]
        for cand in rest:
            if env.pilots_left <= 0:
                break
            run_safe(cand, STAGE1_N)

        for cand in candidates:
            cand["post_raw"], cand["post_std"] = posterior_raw(cand)
            cand["conservative_raw"] = cand["post_raw"] - UNCERTAINTY_K * cand["post_std"]

        # --- 3. Канал + отбор финальных кампаний ---
        for cand in candidates:
            ch, net_per_contact = _best_channel(cand["conservative_raw"], cand["avg_arpu"], channels)
            cand["channel"] = ch
            cand["net_per_contact"] = net_per_contact
            cand["campaign_value"] = net_per_contact * min(cand["n"], MAX_CUSTOMERS_PER_CAMPAIGN)

        finalists = sorted((c for c in candidates if c["net_per_contact"] > 0),
                            key=lambda c: c["campaign_value"], reverse=True)

        campaigns = []
        remaining_budget = env.remaining_budget
        remaining_contacts = env.remaining_contacts
        for cand in finalists:
            if len(campaigns) >= MAX_CAMPAIGNS:
                break
            n = min(cand["n"], MAX_CUSTOMERS_PER_CAMPAIGN, remaining_contacts)
            cost_pc = channels[cand["channel"]]["cost_per_contact"]
            if cost_pc > 0:
                n = min(n, int(remaining_budget // cost_pc))
            if n < 10:
                continue
            campaigns.append({
                "campaign_name": f"{cand['current_tariff']}_{cand['arpu_segment']}_to_{cand['target_tariff']}",
                "filter_arpu_segment": cand["arpu_segment"],
                "filter_current_tariff": cand["current_tariff"],
                "target_tariff": cand["target_tariff"],
                "channel": cand["channel"],
            })
            remaining_contacts -= n
            remaining_budget -= n * cost_pc

        return campaigns
