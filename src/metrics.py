import math
from collections import Counter

class Metrics:
    def __init__(self, k: int, train: list[tuple[int, list[int]]]):
        """
        Считает метрики
        k: размер top-k
        train: список (user_id, [item_id1, item_id2, ...])
        """
        self.k = k
        self.item_freq = Counter()
        self.train_items = set()
        user_ids = set()

        for user_id, items in train:
            user_ids.add(user_id)
            self.train_items.update(items)
            self.item_freq.update(items)

        self.num_users = len(user_ids)

    def compute(self, recommendations: list[dict]) -> dict:
        hitrate_scores = []
        ndcg_scores = []
        coverage_set = set()
        surprisal_scores = []

        for rec in recommendations:
            user_id = rec["user_id"]
            pred_items = rec["topk_items"][:self.k]
            target_item = rec["target"]

            # HitRate@k
            hitrate = int(target_item in pred_items)
            hitrate_scores.append(hitrate)

            # NDCG@k
            if target_item in pred_items:
                rank = pred_items.index(target_item)
                dcg = 1 / math.log2(rank + 2)
            else:
                dcg = 0.0
            idcg = 1.0  # один релевантный элемент наилучший случай
            ndcg = dcg / idcg
            ndcg_scores.append(ndcg)

            # Coverage@k
            coverage_set.update(pred_items)

            # Surprisal@k
            surprisal = 0.0
            for item in pred_items:
                u_j = self.item_freq.get(item, 1)
                info = -math.log2(u_j / self.num_users)
                norm_info = info / math.log2(self.num_users)
                surprisal += norm_info
            surprisal_scores.append(surprisal / self.k)

        return {
            f"HitRate@{self.k}": round(sum(hitrate_scores) / len(hitrate_scores), 6) if hitrate_scores else 0.0,
            f"NDCG@{self.k}": round(sum(ndcg_scores) / len(ndcg_scores), 6) if ndcg_scores else 0.0,
            f"Coverage@{self.k}": round(len(coverage_set) / len(self.train_items), 6) if self.train_items else 0.0,
            f"Surprisal@{self.k}": round(sum(surprisal_scores) / len(surprisal_scores), 6) if surprisal_scores else 0.0,
        }
