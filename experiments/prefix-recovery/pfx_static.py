from lexsi_v2_trainer import LexsiV2Trainer
from pfx_lib import DIR_LAYER
class Static(LexsiV2Trainer):
    """Fixed direction from DIR_LAYER (estimated once at init, single-BOS); random layer subset, weight 1 (DeepRefusal-style); used with recompute_every=0."""
    def _select_active_layers(self):
        NL = len(self._hooks); r = self._rng; k = r.randint(max(1, NL // 6), max(1, NL // 2)); ls = set(r.sample(range(NL), k)); L = min(DIR_LAYER, NL - 1)
        for i, h in enumerate(self._hooks): h.d = self.D[L]; h.w = 1.0 if i in ls else 0.0; h.active = True
        self.attack_log.append((k, sorted(ls)[:3]))
