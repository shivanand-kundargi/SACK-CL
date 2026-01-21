import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import logging


class ConceptUnitsRegularizer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.units = None  # LongTensor [|U|]
        self.weights = None  # FloatTensor [|U|]
        self.mu = None  # FloatTensor [|U|]
        self.std = None  # FloatTensor [|U|]
        self.ready = False

    def load_for_task(self, model, task_idx_1based: int):
        stats_dir = self.args.concept_units_stats_dir or f"SACK/saved_activations_{model.NAME}_{model.dataset.NAME}"
        file_name = f"{task_idx_1based}_{model.NAME}_{model.dataset.NAME}_{self.args.concept_units_layer}.pt"
        pt_path = os.path.join(stats_dir, file_name)
        if not os.path.exists(pt_path):
            logging.warning(f"ConceptUnitsRegularizer: activation file not found: {pt_path}")
            self.ready = False
            return

        feats = torch.load(pt_path, map_location='cpu').float()  # [N, D]
        mu = feats.mean(dim=0)                                   # [D]
        std = feats.std(dim=0) + 1e-6                            # [D]

        units, weights = None, None
        try:
            from utils.concept_paths import PATHS_CONFIG
            csv_path = os.path.join(
                PATHS_CONFIG['activations_path'],
                f"results_task{task_idx_1based}_{model.NAME}_{model.dataset.NAME}",
                "descriptions.csv"
            )
        except Exception:
            csv_path = None

        if csv_path is not None and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df[df['layer'] == self.args.concept_units_layer]
            sims = df['similarity'].to_numpy()
            thr = float(np.percentile(sims, self.args.concept_units_percentile))
            df = df[df['similarity'] >= thr]
            units = torch.tensor(df['unit'].astype(int).tolist(), dtype=torch.long)
            s = torch.tensor(df['similarity'].values, dtype=torch.float32)
            weights = (s - s.min()) / (s.max() - s.min() + 1e-8)
        else:
            D = mu.numel()
            units = torch.arange(D, dtype=torch.long)
            weights = torch.ones(D, dtype=torch.float32)
            logging.warning(f"ConceptUnitsRegularizer: CLIP-Dissect CSV not found, using all units with uniform weights")

        self.units = units.to(self.device)
        self.mu = mu[units].to(self.device)
        self.std = std[units].to(self.device)
        self.weights = weights.to(self.device)
        self.ready = True

    def _extract_features(self, model, inputs):
        # Try Mammoth backbones
        try:
            return model.net(inputs, returnt='features')
        except Exception:
            pass
        # Try coda-prompt
        try:
            return model.net(inputs, pen=True, train=False)
        except Exception:
            pass
        # Try generic .features
        try:
            return model.net.features(inputs)
        except Exception:
            raise RuntimeError("Feature extraction not supported; model needs a custom extractor.")

    def penalty(self, model, inputs):
        if not self.ready or self.args.concept_units_lambda <= 0:
            return None
        feats = self._extract_features(model, inputs)            # [B, D]
        # import pdb; pdb.set_trace()
        f_sel = feats.index_select(1, self.units)                # [B, |U|]
        diff = (f_sel - self.mu) / self.std
        w = self.weights.unsqueeze(0)                            # [1, |U|]
        return (w * (diff ** 2)).mean()


