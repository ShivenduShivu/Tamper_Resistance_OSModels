"""LexsiV2Trainer: the reference defence v2 objective expressed on SafeTune's DeepRefusalTrainer (safetune==0.1.12, safetune.interventions.harden.deeprefusal).

Reference objective (kaggle/day3/day3_defense_v2.py): single mixed SFT cross-entropy over extended-refusal + benign rows; on a micro-step with probability P_ATTACK a
Heretic-like simulated abliteration is applied in the forward pass: one unit direction D[src] (src ~ U[NL/4, 3NL/4]) is projected out of EVERY layer's output with a per-layer weight
w_i = top * max(0, 1 - |i - peak| / width) (peak ~ U[NL/4, 3NL/4], width ~ U[NL/6, NL/2], top ~ U[0.7, 1]); the per-layer directions D (mean harmful - mean benign last-prompt-token
hidden state per layer, unit norm) are re-estimated from the CURRENT model every `recompute_every` optimizer steps.

What is reused from SafeTune: HF Trainer subclass with peft LoRA wrapping (r/alpha/dropout/target_modules), the forward-hook installation point (_install_hooks), hook activation
interface (_select_active_layers / _deactivate_all_hooks), _forward_ce, post-training merge (_maybe_merge_lora).
What is overridden (precise API gaps in DeepRefusalTrainer vs the reference objective):
  1. direction: DeepRefusalTrainer takes ONE fixed 1-D direction; reference uses per-layer directions refreshed during training  -> per-layer matrix D + refresh callback.
  2. ablation placement: DeepRefusalTrainer ablates a random subset of layers with Bernoulli(p) per layer each at full strength; reference applies one direction to all layers with a peaked weight profile -> weighted hook.
  3. loss: DeepRefusalTrainer = alpha*L_harmful(ablated, own iterator) + (1-alpha)*L_benign(not ablated); reference = one mixed CE where the ablation applies to the whole micro-batch (benign rows too) -> compute_loss override.
  4. defaults: LoRA r16/alpha32 vs reference r32/alpha64 (set via config); scheduler: OneCycleLR(pct_start=0.05) reproduced by create_scheduler override; optimizer: reference torch AdamW default weight_decay=0.01 (HF default 0.0) -> set weight_decay=0.01; grad clip 1.0 both; loss scaled mean/ACC (explicit, see compute_loss).
  Remaining known non-identities: (i) HF steps the optimizer at the end of each epoch on the leftover micro-batches (2355 rows -> 295 steps/epoch, 590 total) with the divisor still ACC, so that short window is down-weighted; the reference carries leftovers into the next epoch and runs 588 steps (OneCycle total_steps differs by 2); (ii) Trainer data order/seed differs from random.shuffle; direction-estimation sampling uses its own seeded RNG (same distribution); HF fp16 autocast+GradScaler vs torch's (same mechanism).
Not changed: LoRA wrapping, merge, hook mechanism, training loop (HF Trainer, grad accumulation)."""
from __future__ import annotations
import random
from typing import Any, List, Optional
import torch
from transformers import TrainerCallback
from safetune.interventions.harden.deeprefusal import DeepRefusalTrainer, DeepRefusalConfig, _get_decoder_layers

__all__ = ["LexsiV2Config", "LexsiV2Trainer", "WeightedAblationHook"]

class LexsiV2Config(DeepRefusalConfig):
    pass

class WeightedAblationHook:
    """h <- h - w * (h . d) d on a decoder layer's output while `active`; identity otherwise."""
    def __init__(self) -> None:
        self.active = False; self.d: Optional[torch.Tensor] = None; self.w = 0.0
    def __call__(self, module, inputs, outputs):
        if not self.active or self.d is None or self.w == 0.0: return outputs
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        d = self.d.to(device=h.device, dtype=h.dtype)
        h = h - self.w * (h @ d).unsqueeze(-1) * d
        return (h,) + tuple(outputs[1:]) if isinstance(outputs, tuple) else h

class _RefreshCallback(TrainerCallback):
    def __init__(self, trainer): self.t = trainer
    def on_step_end(self, args, state, control, **kw):
        if self.t.recompute_every and state.global_step % self.t.recompute_every == 0:
            self.t.refresh_directions(state.global_step)

class LexsiV2Trainer(DeepRefusalTrainer):
    def __init__(self, *args, tokenizer, harmful_prompts: List[str], benign_prompts: List[str], p_attack: float = 0.5, recompute_every: int = 100,
                 n_dir_prompts: int = 96, dir_batch: int = 4, direction_add_special_tokens: bool = True, chat_template_fn=None, seed: int = 0, **kwargs):
        self.tok = tokenizer; self.harm_p = harmful_prompts; self.ben_p = benign_prompts; self.p_attack = p_attack; self.recompute_every = recompute_every
        self.n_dir = n_dir_prompts; self.dir_bs = dir_batch; self.dir_add_special = direction_add_special_tokens
        self.fmt = chat_template_fn or (lambda p: self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))
        self._rng = random.Random(seed); random.seed(seed)
        self.refresh_log: list = []; self.attack_log: list = []
        hidden = kwargs.get("model", args[0] if args else None).config.hidden_size
        super().__init__(*args, refusal_direction=torch.ones(hidden), **kwargs)   # placeholder direction; per-layer D is estimated below
        if not hasattr(self, "loss_is_scaled_for_ga"): raise RuntimeError("this transformers version has no Trainer.loss_is_scaled_for_ga; gradient-accumulation scaling would be applied twice")
        self.NL = len(self._hooks); self.D: Optional[torch.Tensor] = None
        self.refresh_directions(0)
        self.add_callback(_RefreshCallback(self))

    # --- hook installation (override of the SafeTune extension point): weighted per-layer hooks ---
    def _install_hooks(self) -> None:
        layers = _get_decoder_layers(self.model)
        self._hooks, self._hook_handles = [], []
        for layer in layers:
            h = WeightedAblationHook(); self._hooks.append(h); self._hook_handles.append(layer.register_forward_hook(h))

    def _select_active_layers(self) -> None:           # SafeTune interface; reference semantics (Heretic-like profile, one direction)
        NL = len(self._hooks); r = self._rng
        src = r.randint(NL // 4, 3 * NL // 4); peak = r.randint(NL // 4, 3 * NL // 4)
        width = r.uniform(NL / 6, NL / 2); top = r.uniform(0.7, 1.0)
        for i, h in enumerate(self._hooks):
            h.d = self.D[src]; h.w = top * max(0.0, 1 - abs(i - peak) / width); h.active = True
        self.attack_log.append((src, peak, round(width, 3), round(top, 3)))

    @torch.no_grad()
    def refresh_directions(self, step: int) -> None:
        model = self.model; was = model.training; pad = self.tok.padding_side
        hook_state = [(h.active) for h in self._hooks]; self._deactivate_all_hooks()   # directions are always estimated with the ablation off
        try:
            model.eval(); self.tok.padding_side = "left"; acc = {}
            for name, pool in (("h", self.harm_p), ("b", self.ben_p)):
                ps = self._rng.sample(pool, min(self.n_dir, len(pool))); s = None
                for i in range(0, len(ps), self.dir_bs):
                    enc = self.tok([self.fmt(p) for p in ps[i:i + self.dir_bs]], return_tensors="pt", padding=True, truncation=True, max_length=1024,
                                   add_special_tokens=self.dir_add_special).to(next(model.parameters()).device)
                    hs = model(**enc, output_hidden_states=True).hidden_states[1:]
                    v = torch.stack([h[:, -1, :].float().sum(0) for h in hs]); s = v if s is None else s + v
                acc[name] = s / len(ps)
            d = acc["h"] - acc["b"]; nrm = d.norm(dim=-1, keepdim=True)
            if not torch.isfinite(d).all() or (nrm < 1e-6).any():
                self.refresh_log.append({"step": step, "skipped": "non-finite or ~zero-norm direction; previous directions kept"})
                if self.D is None: raise RuntimeError("initial refusal directions are degenerate")
                return
            d = d / nrm; prev = self.D; self.D = d.detach().cpu()
            self.refresh_log.append({"step": step, "cos_prev": None if prev is None else round((prev * self.D).sum(-1).mean().item(), 4)})
        finally:                                        # restore on success AND on exception
            model.train(was); self.tok.padding_side = pad
            for h, a_ in zip(self._hooks, hook_state): h.active = a_

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self._rng.random() < self.p_attack: self._select_active_layers()
        loss = self._forward_ce(model, inputs)   # one mixed CE; ablation (if active) applies to the whole micro-batch
        # reference: loss = CE_microbatch_mean / ACC, accumulated over ACC micro-batches. The Trainer would instead expect a token-count-normalised loss
        # (num_items_in_batch) when the model accepts loss kwargs, so the scaling is made explicit here and the Trainer's own scaling is disabled.
        self.loss_is_scaled_for_ga = True
        loss = loss / self.args.gradient_accumulation_steps
        return (loss, None) if return_outputs else loss

    def create_scheduler(self, num_training_steps, optimizer=None):
        # reference: torch OneCycleLR(max_lr=lr, total_steps=steps+1, pct_start=0.05) stepped once per optimizer step (default cos anneal, momentum cycling on)
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer or self.optimizer, max_lr=self.args.learning_rate, total_steps=num_training_steps + 1, pct_start=0.05)
        return self.lr_scheduler

    def training_step(self, *a, **k):
        # hooks must stay active through backward (gradient checkpointing recomputes the forward there); always cleared afterwards
        try: return super().training_step(*a, **k)
        finally: self._deactivate_all_hooks()
