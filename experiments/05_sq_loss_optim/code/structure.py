r"""Merging and splitting of SQs during optimization (config section "structure").

Notation: claims w_pj, strong set S_j = {p : w_pj > strong_claim}, voted label l_j = argmax_k votes_j(k),
purity pi_j = max_k votes_j(k).

Merge:  co-ownership C_ij = |S_i & S_j| / min(|S_i|, |S_j|) (overlap coefficient), smoothed over checks
        C^_ij <- ema * C^_ij + (1 - ema) * C_ij. Merge if l_i == l_j and C^_ij > threshold; pairs greedily by
        descending C^, every SQ in at most one merge per check. (i, j) -> fit(S_i | S_j),
        eps = mean, existence = max.
Split:  if pi_j < purity_threshold and |S_j| >= min_points: with k* the majority GT instance in S_j,
        A = {p in S_j : y_p = k*}, B = S_j \ A; j -> fit(A), fit(B) (both keep eps and existence).
        At most max_fraction * N splits per check (lowest purity first), total SQs <= max_sqs.
fit(P): t = mean, R = principal axes (det +1), a_k = quantile_q |R_k^T (x - t)| + margin.
New SQs get fresh Adam state; kept SQs keep theirs.
"""

import math

import torch

from optimize import DEVICE, claims, sdf_matrix


def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(M, 3, 3) rotation matrices -> (M, 4) quaternions (w, x, y, z)."""
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cands = torch.stack([
        torch.stack([1 + t, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], 1),
        torch.stack([m[:, 2, 1] - m[:, 1, 2], 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0]], 1),
        torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 0, 1] + m[:, 1, 0], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], m[:, 1, 2] + m[:, 2, 1]], 1),
        torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1], 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], 1),
    ], 1)  # (M, 4 candidates, 4); pick the numerically largest one
    diag = torch.stack([1 + t, 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                        1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], 1)
    q = cands[torch.arange(len(m)), diag.argmax(1)]
    return torch.nn.functional.normalize(q, dim=1)


def fit(points: torch.Tensor, cfg_fit: dict, scale_min: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Principal-axes box fit of a point set -> (translation (3,), rotation (3, 3) local-to-world, scale (3,))."""
    mu = points.mean(0)
    x = points - mu
    _, vecs = torch.linalg.eigh(x.T @ x / max(len(x) - 1, 1))
    R = vecs.flip(1)  # largest axis first
    if torch.det(R) < 0:
        R[:, 2] = -R[:, 2]
    proj = (x @ R).abs()
    a = torch.quantile(proj, cfg_fit["quantile"], dim=0) + cfg_fit["margin"]
    return mu, R, a.clamp_min(scale_min)


class Structure:
    def __init__(self, cfg: dict, n: int):
        self.cfg = cfg
        self.s = cfg["structure"]
        self.coown = torch.zeros(n, n, device=DEVICE)  # smoothed co-ownership C^
        self.max_sqs = int(self.s["max_sqs_factor"] * n)

    def due(self, step: int) -> bool:
        s = self.s
        return s["enabled"] and s["start"] <= step <= s["stop"] and (step - s["start"]) % s["every"] == 0

    @torch.no_grad()
    def statistics(self, sq: dict, scene: dict, chunk: int = 131072):
        """One pass over all points: co-ownership counts, strong-set sizes, votes."""
        x, y, K = scene["pos"], scene["y"], len(scene["instance_ids"])
        n = len(sq["trans"])
        inter = torch.zeros(n, n, device=DEVICE)
        vote_num = torch.zeros(n, K, device=DEVICE)
        for i in range(0, len(x), chunk):
            w = claims(self.cfg, sq, sdf_matrix(self.cfg, sq, x[i:i + chunk]))
            strong = (w > self.s["strong_claim"]).float()
            inter += strong @ strong.T
            vote_num.index_add_(1, y[i:i + chunk], w)
        votes = vote_num / (vote_num.sum(1, keepdim=True) + 1e-6)
        return inter, inter.diag(), votes

    @torch.no_grad()
    def strong_points(self, sq: dict, scene: dict, ids: list[int], chunk: int = 131072) -> dict[int, torch.Tensor]:
        """Indices of the strong set S_j for the given SQs."""
        x = scene["pos"]
        sub = {k: v[ids] for k, v in sq.items()}
        out = [[] for _ in ids]
        for i in range(0, len(x), chunk):
            w = claims(self.cfg, sub, sdf_matrix(self.cfg, sub, x[i:i + chunk]))
            for r in range(len(ids)):
                out[r].append(torch.nonzero(w[r] > self.s["strong_claim"]).squeeze(1) + i)
        return {j: torch.cat(o) for j, o in zip(ids, out)}

    @torch.no_grad()
    def step(self, model, opt, scene: dict) -> dict:
        s, sq = self.s, model()
        n = len(sq["trans"])
        inter, size, votes = self.statistics(sq, scene)
        label, purity = votes.argmax(1), votes.max(1).values
        c = inter / torch.minimum(size[:, None], size[None, :]).clamp_min(1)
        c.fill_diagonal_(0)
        self.coown = s["merge"]["ema"] * self.coown + (1 - s["merge"]["ema"]) * c

        used = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        merges = []
        if s["merge"]["enabled"]:
            cand = torch.triu(self.coown, 1) > s["merge"]["threshold"]
            if s["merge"]["same_label"]:
                cand &= label[:, None] == label[None, :]
            ii, jj = torch.nonzero(cand, as_tuple=True)
            for k in torch.argsort(self.coown[ii, jj], descending=True).tolist():
                i, j = ii[k].item(), jj[k].item()
                if not used[i] and not used[j]:
                    merges.append((i, j))
                    used[i] = used[j] = True

        splits = []
        if s["split"]["enabled"]:
            budget = min(int(s["split"]["max_fraction"] * n), self.max_sqs - (n - len(merges)))
            cand = (purity < s["split"]["purity_threshold"]) & (size >= s["split"]["min_points"]) & ~used
            for j in torch.nonzero(cand).squeeze(1)[torch.argsort(purity[cand])].tolist()[:max(budget, 0)]:
                splits.append(j)

        if not merges and not splits:
            return {"struct/merges": 0, "struct/splits": 0, "struct/n_sqs": n}

        pts = self.strong_points(sq, scene, sorted({i for m in merges for i in m} | set(splits)))
        new = []  # (t, R, a, eps, existence)
        removed = set()
        n_merge = 0
        for i, j in merges:
            P = scene["pos"][torch.cat([pts[i], pts[j]]).unique()]
            if len(P) < 4:
                continue
            t, R, a = fit(P, s["fit"], model.scale_min)
            new.append((t, R, a, (sq["eps"][i] + sq["eps"][j]) / 2, torch.maximum(sq["existence"][i], sq["existence"][j])))
            removed |= {i, j}
            n_merge += 1
        n_split = 0
        for j in splits:
            idx = pts[j]
            yj = scene["y"][idx]
            major = torch.bincount(yj).argmax()
            parts = [idx[yj == major], idx[yj != major]]
            if min(len(p) for p in parts) < s["split"]["min_part_points"]:
                continue
            for p in parts:
                t, R, a = fit(scene["pos"][p], s["fit"], model.scale_min)
                new.append((t, R, a, sq["eps"][j], sq["existence"][j]))
            removed.add(j)
            n_split += 1

        keep = torch.tensor([k for k in range(n) if k not in removed], device=DEVICE, dtype=torch.long)
        self._resize(model, opt, keep, new)
        m = len(keep) + len(new)
        coown = torch.zeros(m, m, device=DEVICE)
        coown[:len(keep), :len(keep)] = self.coown[keep][:, keep]
        self.coown = coown
        return {"struct/merges": n_merge, "struct/splits": n_split, "struct/n_sqs": m}

    def _resize(self, model, opt, keep: torch.Tensor, new: list):
        """Keep the given SQ rows, append new SQs; carry Adam state for kept rows, zeros for new ones."""
        if new:
            t = torch.stack([x[0] for x in new])
            q = matrix_to_quat(torch.stack([x[1] for x in new]))
            log_a = torch.stack([x[2] for x in new]).clamp(model.scale_min, model.scale_max).log()
            eps = torch.stack([x[3] for x in new])
            eps_raw = torch.logit(((eps - model.eps_min) / (model.eps_max - model.eps_min)).clamp(1e-4, 1 - 1e-4))
            ex_logit = torch.logit(torch.stack([x[4] for x in new]).clamp(1e-4, 1 - 1e-4))
            fresh = {"translation": t, "log_scale": log_a, "eps_raw": eps_raw, "rotation": q, "existence_logit": ex_logit}
        for name in ("translation", "log_scale", "eps_raw", "rotation", "existence_logit"):
            old = getattr(model, name)
            data = old.data[keep]
            if new:
                data = torch.cat([data, fresh[name].to(data.dtype)])
            p = torch.nn.Parameter(data)
            for group in opt.param_groups:
                if group["params"][0] is old:
                    group["params"] = [p]
            st = opt.state.pop(old, None)
            if st:
                pad = lambda v: torch.cat([v[keep], torch.zeros((len(new),) + v.shape[1:], device=v.device, dtype=v.dtype)])  # noqa: E731
                opt.state[p] = {"step": st["step"], "exp_avg": pad(st["exp_avg"]), "exp_avg_sq": pad(st["exp_avg_sq"])}
            setattr(model, name, p)
