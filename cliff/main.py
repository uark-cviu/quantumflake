import os
import argparse
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.nn.functional as F

from vision_transformer import vit_base_patch16_224
from prompt import Prompt
from dataset import MaterialDataset
from clifff_head import cliffFHead

THICKNESS = ["Few", "Mono", "Thick"]


def seed_everything(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class MemoryBank:
    def __init__(self, capacity_per_class=50):
        self.capacity_per_class = capacity_per_class
        self.memory = defaultdict(list)

    def add_samples(self, dataset, class_names):
        for sample_path, label_idx in dataset.samples:
            class_name = class_names[label_idx]
            if len(self.memory[class_name]) < self.capacity_per_class:
                self.memory[class_name].append((sample_path, label_idx))
            else:
                if random.random() < 0.1:
                    idx = random.randint(0, len(self.memory[class_name]) - 1)
                    self.memory[class_name][idx] = (sample_path, label_idx)

    def get_memory_dataset(self, transform=None):
        all_samples = []
        for class_samples in self.memory.values():
            all_samples.extend(class_samples)

        if not all_samples:
            return None

        return MemoryDataset(all_samples, transform)


class MemoryDataset:
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def knowledge_distillation_loss(student_logits, teacher_logits, temperature=3.0):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)


def build_order_and_maps(global_class_map: Dict[str, int], seen_materials: List[str]) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    desired = []
    for m in seen_materials:
        for t in THICKNESS:
            name = f"{t}_{m}"
            if name in global_class_map:
                desired.append(name)
    inv_order = {name: i for i, name in enumerate(desired)}
    idx_map = []
    id_to_mat = []
    mat_to_id = {m: i for i, m in enumerate(seen_materials)}
    for name, ds_idx in sorted(global_class_map.items(), key=lambda kv: kv[1]):
        if name in inv_order:
            idx_map.append(inv_order[name])
            mat = name.split('_')[1]
            id_to_mat.append(mat_to_id[mat])

    return desired, torch.tensor(idx_map, dtype=torch.long), torch.tensor(id_to_mat, dtype=torch.long)


def train_one_epoch_improved(model, prompt, cliff, data_loader, memory_loader, criterion, optimizer,
                           device, idx_map, id_to_mat, gate_w=0.1, kd_weight=0.5,
                           old_model=None, scale_delta_by_gate=False, new_mid=None):
    model.train(); prompt.train(); cliff.train()
    if old_model is not None:
        old_model.eval()

    total_loss, correct, n = 0.0, 0, 0
    ce = criterion
    data_iter = iter(data_loader)
    memory_iter = iter(memory_loader) if memory_loader else None

    max_steps = max(len(data_loader), len(memory_loader) if memory_loader else 0)

    for step in range(max_steps):
        optimizer.zero_grad()
        total_step_loss = 0.0
        step_samples = 0
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            images, labels = next(data_iter)

        images, labels = images.to(device), labels.to(device)

        feats = model(images, prompt=prompt)['pre_logits']
        logits_3m, sims = cliff.logits_3M(feats, scale_delta_by_gate=scale_delta_by_gate)
        logits = logits_3m[:, idx_map.to(device)]
        current_loss = ce(logits, labels)
        total_step_loss += current_loss
        if gate_w > 0.0 and sims is not None and sims.numel() > 0:
            mat_ids = id_to_mat.to(device)[labels]
            gate_loss = gate_w * ce(sims, mat_ids)
            total_step_loss += gate_loss

        step_samples += images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        if memory_iter is not None:
            try:
                mem_images, mem_labels = next(memory_iter)
            except StopIteration:
                memory_iter = iter(memory_loader)
                mem_images, mem_labels = next(memory_iter)

            mem_images, mem_labels = mem_images.to(device), mem_labels.to(device)

            mem_feats = model(mem_images, prompt=prompt)['pre_logits']
            mem_logits_3m, mem_sims = cliff.logits_3M(mem_feats, scale_delta_by_gate=scale_delta_by_gate)
            mem_logits = mem_logits_3m[:, idx_map.to(device)]
            memory_loss = ce(mem_logits, mem_labels)
            total_step_loss += memory_loss
            if old_model is not None and kd_weight > 0:
                with torch.no_grad():
                    old_feats = old_model(mem_images, prompt=prompt)['pre_logits']
                    old_logits_3m, _ = cliff.logits_3M(old_feats, scale_delta_by_gate=scale_delta_by_gate)
                    old_logits = old_logits_3m[:, idx_map.to(device)]

                kd_loss = kd_weight * knowledge_distillation_loss(mem_logits, old_logits)
                total_step_loss += kd_loss

            step_samples += mem_images.size(0)
            correct += (mem_logits.argmax(dim=1) == mem_labels).sum().item()

        total_step_loss.backward()
        if new_mid is not None and cliff.material_embedding.weight.grad is not None:
            g = cliff.material_embedding.weight.grad
            mask = torch.zeros_like(g, dtype=torch.bool); mask[new_mid] = True
            g[~mask] = 0.0
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(prompt.parameters()) + list(cliff.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += total_step_loss.item() * step_samples
        n += step_samples

    return total_loss / max(1, n), 100.0 * correct / max(1, n)



@torch.no_grad()
def evaluate_cliff(model, prompt, cliff, data_loader, criterion, device, idx_map, scale_delta_by_gate=False):
    model.eval(); prompt.eval(); cliff.eval()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        feats = model(images, prompt=prompt)['pre_logits']
        logits_3m, _ = cliff.logits_3M(feats, scale_delta_by_gate=scale_delta_by_gate)
        logits = logits_3m[:, idx_map.to(device)]
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        n += labels.size(0)
    return total_loss / max(1, n), 100.0 * correct / max(1, n)


def print_task_table(task_id: int, seen_materials: List[str], acc_hist: DefaultDict[str, List[float]]):
    header = "Task {:>2d} | ".format(task_id + 1) + " | ".join(f"{m:>9s}" for m in seen_materials)
    row = "          | " + " | ".join(f"{(acc_hist[m][-1] if acc_hist[m] else float('nan')):>9.2f}" for m in seen_materials)
    print(header); print(row)


def make_prompt(embed_dim: int, args) -> Prompt:
    return Prompt(
        length=args.prompt_length,
        embed_dim=embed_dim,
        embedding_key="mean",
        prompt_init="uniform",
        prompt_pool=True,
        prompt_key=True,
        pool_size=args.prompt_pool_size,
        top_k=args.prompt_top_k,
        batchwise_prompt=True
    ).to(args.device if torch.cuda.is_available() else "cpu")


def copy_model(model):
    model_copy = type(model)(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm
    )
    model_copy.load_state_dict(model.state_dict())
    return model_copy.eval()


def main(args):
    save_path = "/content/drive/MyDrive/clifff_model.pth"
    print(f"Save Path: {save_path}")
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = vit_base_patch16_224(pretrained=True).to(device)
    d_z = getattr(model, "embed_dim", getattr(model, "num_features", 768))
    cliff = cliffFHead(
        d_z=d_z,
        d_e=args.material_embed_dim,
        gate_temp=args.gate_temp,
        init_materials=[args.materials[0]]
    ).to(device)
    memory_bank = MemoryBank(capacity_per_class=args.memory_size)
    prompts: Dict[str, Prompt] = {}

    seen_materials: List[str] = []
    global_class_map: Dict[str, int] = {}
    acc_hist: DefaultDict[str, List[float]] = defaultdict(list)
    old_model = None

    for task_id, material in enumerate(args.materials):
        print("\n" + "=" * 60)
        print(f"TASK {task_id + 1}/{len(args.materials)}: Train on {material} (WITH MEMORY REPLAY)")
        print("=" * 60)
        mid = cliff.ensure_material(material)
        base_param = next(cliff.base_head.parameters())
        cliff.material_fcs[f"material_{mid}"].to(device=base_param.device, dtype=base_param.dtype)

        if material not in prompts:
            prompts[material] = make_prompt(d_z, args)
        prompt_cur = prompts[material]
        curr_train_root = os.path.join(args.data_path, material, 'train')
        if not os.path.isdir(curr_train_root):
            raise FileNotFoundError(f"Missing: {curr_train_root}")
        for class_name in sorted(os.listdir(curr_train_root)):
            if class_name not in global_class_map:
                global_class_map[class_name] = len(global_class_map)

        seen_materials.append(material)
        train_dataset = MaterialDataset(
            root_dir=args.data_path,
            materials=[material],
            split='train',
            class_to_idx=global_class_map,
            transform=transform_train
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

        class_names = {v: k for k, v in global_class_map.items()}
        memory_bank.add_samples(train_dataset, class_names)
        memory_dataset = memory_bank.get_memory_dataset(transform=transform_train)
        memory_loader = None
        if memory_dataset is not None and task_id > 0:
            memory_loader = DataLoader(memory_dataset, batch_size=args.batch_size//2,
                                     shuffle=True, num_workers=args.num_workers, pin_memory=True)

        _, idx_map, id_to_mat = build_order_and_maps(global_class_map, seen_materials)
        idx_map = idx_map.to(device); id_to_mat = id_to_mat.to(device)

        criterion = nn.CrossEntropyLoss()
        if task_id == 0:
            for p in model.parameters(): p.requires_grad = True
            for p in cliff.base_head.parameters(): p.requires_grad = True
            for p in cliff.gate_proj.parameters(): p.requires_grad = True
            for p in cliff.material_embedding.parameters(): p.requires_grad = False
            for m in cliff.material_fcs.values():
                for p in m.parameters(): p.requires_grad = False
            optimizer = optim.Adam([
                {'params': model.parameters(), 'lr': args.lr * 0.1},
                {'params': cliff.base_head.parameters(), 'lr': args.lr},
                {'params': cliff.gate_proj.parameters(), 'lr': args.lr},
                {'params': prompt_cur.parameters(), 'lr': args.lr_prompt_inc},
            ])
            gate_w = 0.0; new_mid = None
        else:
            for p in model.parameters(): p.requires_grad = False
            for p in cliff.base_head.parameters(): p.requires_grad = False
            for p in cliff.gate_proj.parameters(): p.requires_grad = False
            for m_idx in range(cliff.num_materials):
                head_i = cliff.material_fcs[f"material_{m_idx}"]
                req = (m_idx == mid)
                for p in head_i.parameters(): p.requires_grad = req
            for p in cliff.material_embedding.parameters(): p.requires_grad = True

            optimizer = optim.Adam([
                {'params': prompt_cur.parameters(), 'lr': args.lr_prompt_inc},
                {'params': cliff.material_fcs[f"material_{mid}"].parameters(), 'lr': args.lr},
                {'params': [cliff.material_embedding.weight], 'lr': args.lr_emb},
            ])
            gate_w = args.gate_loss_weight; new_mid = mid
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch_improved(
                model, prompt_cur, cliff, train_loader, memory_loader, criterion, optimizer, device,
                idx_map=idx_map, id_to_mat=id_to_mat, gate_w=gate_w, kd_weight=args.kd_weight,
                old_model=old_model, scale_delta_by_gate=args.scale_delta_by_gate_train, new_mid=new_mid
            )
            print(f"  Epoch {epoch+1:02d}/{args.epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        if task_id == 0:
            old_model = copy_model(model).to(device)
        for p in prompt_cur.parameters(): p.requires_grad = False
        print("\nPer-material validation (seen so far):")
        for m_eval in seen_materials:
            prompt_eval = prompts[m_eval]

            val_dataset = MaterialDataset(
                root_dir=args.data_path,
                materials=[m_eval],
                split='val',
                class_to_idx=global_class_map,
                transform=transform_val
            )
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)

            _, idx_map_eval, _ = build_order_and_maps(global_class_map, seen_materials)
            val_loss, val_acc = evaluate_cliff(
                model, prompt_eval, cliff, val_loader, criterion, device, idx_map_eval.to(device),
                scale_delta_by_gate=args.scale_delta_by_gate_eval
            )
            acc_hist[m_eval].append(val_acc)
            print(f"  {m_eval:>9s}: Val Acc = {val_acc:6.2f}%  (loss {val_loss:.4f})")

        print()
        print_task_table(task_id, seen_materials, acc_hist)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    final_accs = []
    for m in args.materials:
        if acc_hist[m]:
            final_accs.append(acc_hist[m][-1])
            print(f"  Final Acc [{m:>9s}] = {acc_hist[m][-1]:6.2f}%")

    if final_accs:
        print(f"\n  Final Macro-Avg Accuracy: {sum(final_accs)/len(final_accs):.2f}%")

    if len(args.materials) >= 2:
        forget_vals = []
        for m in args.materials[:-1]:
            if acc_hist[m]:
                fm = max(acc_hist[m]) - acc_hist[m][-1]
                forget_vals.append(fm)
                print(f"  Forgetting [{m:>9s}] = max({max(acc_hist[m]):.2f}) - final({acc_hist[m][-1]:.2f}) = {fm:.2f}")
        if forget_vals:
            print(f"\n  Average Forgetting (excl. last): {sum(forget_vals)/len(forget_vals):.2f}%")

    print("\nDone.")
    print(f"\nSaving final model state to {save_path}")
    prompt_states = {name: p.state_dict() for name, p in prompts.items()}
    torch.save({
        'backbone_state_dict': model.state_dict(),
        'clifff_head_state_dict': cliff.state_dict(),
        'prompts_state_dict': prompt_states,
        'global_class_map': global_class_map,
        'seen_materials': seen_materials,
    }, save_path)
    print("Model saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Improved cliff with Memory Replay and Knowledge Distillation')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/data_crops/data_crops')
    parser.add_argument('--materials', nargs='+', default=['BN', 'Graphene', 'MoS2', 'WTe2'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the final model checkpoint')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_emb', type=float, default=1e-4)
    parser.add_argument('--lr_prompt_inc', type=float, default=1e-4)
    parser.add_argument('--memory_size', type=int, default=100)
    parser.add_argument('--kd_weight', type=float, default=1.0)
    parser.add_argument('--gate_temp', type=float, default=1.0)
    parser.add_argument('--gate_loss_weight', type=float, default=0.1)
    parser.add_argument('--scale_delta_by_gate_train', action='store_true', default=False)
    parser.add_argument('--scale_delta_by_gate_eval', action='store_true', default=False)
    parser.add_argument('--material_embed_dim', type=int, default=128)
    parser.add_argument('--prompt_pool_size', type=int, default=30)
    parser.add_argument('--prompt_length', type=int, default=8)
    parser.add_argument('--prompt_top_k', type=int, default=1)
    args = parser.parse_args(args=[])
    main(args)