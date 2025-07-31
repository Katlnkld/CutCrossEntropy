import torch
from tqdm import tqdm

def evaluate(model, dataloader, device, k=10):
    """
    Выполняет предсказания модели для evaluation.

    :param model: обученная модель SASRec
    :param dataloader: test DataLoader
    :param device: 'cuda' или 'cpu'
    :param k: количество top-k рекомендаций

    :return: список словарей с предсказаниями для каждого пользователя
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            user_ids, seqs, targets = batch
            seqs = seqs.to(device)
            targets = targets.to(device)

            # Прямой проход
            logits = model(seqs)                   # (B, T, D)
            final_output = logits[:, -1, :]        # (B, D)
            scores = final_output @ model.item_embedding.weight.T  # (B, V)

            # Top-k предсказания
            topk_scores, topk_indices = torch.topk(scores, k, dim=1)

            for uid, top_items, top_scores, target in zip(user_ids, topk_indices, topk_scores, targets):
                results.append({
                    "user_id": uid.item() if torch.is_tensor(uid) else uid,
                    "topk_items": top_items.tolist(),
                    "topk_scores": top_scores.tolist(),
                    "target": target.item()
                })

    return results
