"""Training loop for STL and MTL with the paper hyperparameters."""
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


def _unique(values):
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def train_epoch(model, loader, optimizer, scheduler, device, mode, max_grad_norm, criteria):
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        tasks = batch["task"]

        if mode == "stl":
            logits = model(input_ids, attention_mask)
            task = tasks[0]
            loss = criteria[task](logits, labels)
        else:
            losses = []
            for task in _unique(tasks):
                idxs = [i for i, item in enumerate(tasks) if item == task]
                logits = model(input_ids[idxs], attention_mask[idxs], task)
                raw = criteria[task](logits, labels[idxs])
                losses.append(model.uncertainty_loss(task, raw))
            loss = torch.stack(losses).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(len(loader), 1)


def evaluate(model, loader, device, mode, section_classes, label2id):
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].tolist()
            tasks = batch["task"]
            names = batch["label_name"]

            def record(i, task, pred):
                return {
                    "instance_id": batch["instance_id"][i],
                    "study_id": batch["study_id"][i],
                    "origin": batch["origin"][i],
                    "task": task,
                    "true_name": names[i],
                    "pred_name": section_classes[task][pred],
                    "true_id": labels[i],
                    "pred_id": pred,
                }

            if mode == "stl":
                preds = torch.argmax(model(input_ids, attention_mask), dim=1).tolist()
                for i, pred in enumerate(preds):
                    records.append(record(i, tasks[i], pred))
            else:
                for task in _unique(tasks):
                    idxs = [i for i, item in enumerate(tasks) if item == task]
                    preds = torch.argmax(
                        model(input_ids[idxs], attention_mask[idxs], task), dim=1
                    ).tolist()
                    for local, pred in enumerate(preds):
                        records.append(record(idxs[local], task, pred))
    return records


def make_scheduler(optimizer, n_steps, warmup_ratio):
    warmup = max(1, int(n_steps * warmup_ratio))
    return get_linear_schedule_with_warmup(optimizer, warmup, n_steps)


def make_loader(dataset, batch_size, shuffle, seed=None):
    """A seeded generator makes the shuffling order repeat across runs."""
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
