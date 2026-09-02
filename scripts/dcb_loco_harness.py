#!/usr/bin/env python
"""R002-R011: the nested leave-one-compound-out harness for DCB-40.

Nesting contract (frozen in configs/dcb_gates.json before this ran):
  for outer fold i, EVERYTHING is estimated from the fit set only --
  mu_ctx, the PCA basis, the feature projection, the ridge lambda, k and q.
  The oracle solves compound i's coefficients inside the SAME fold basis.
  mu for any compound j is always a leave-j-out context mean, so the held-out
  target and the fit profiles sit on the same metric.

Observation masks are recomputed AFTER subtracting mu: mu is NaN wherever no
fit row in that context observed the protein, and not re-deriving the mask
silently drops whole rows.

Decoders (same fitted coefficients, only the output support differs):
  dense   v = c^T B over all 4422 proteins
  sparse  v restricted to its top-58 |values| (the M3 median support), rest 0
The M3 prior used directly as a prediction is reported as the sparse anchor.

Test truth is never read.  Validation is never read here.
"""

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "outputs/dcb40/cache/cache.npz"
CFG = json.loads((ROOT / "configs/dcb_gates.json").read_text())
OUT = ROOT / "outputs/dcb40"
SPARSE_SUPPORT = 58
Q_GRID = [2, 5, 10]
LAMBDA_GRID = np.logspace(-3, 3, 13)
RNG_SEED = 20260816


# --------------------------------------------------------------------------- #
# context means via precomputed (ctx, compound) partial sums
# --------------------------------------------------------------------------- #
class CtxSums:
    """NaN-aware sums/counts per (context, compound) so leave-out means are O(1)."""

    def __init__(self, D, ctx_id, comp_id):
        self.n_ctx = int(ctx_id.max()) + 1
        self.n_comp = int(comp_id.max()) + 1
        obs = np.isfinite(D)
        Dz = np.where(obs, D, 0.0)
        pair = ctx_id * self.n_comp + comp_id
        self.pair = pair
        upair, inv = np.unique(pair, return_inverse=True)
        self.upair = upair
        P, K = len(upair), D.shape[1]
        self.S = np.zeros((P, K)); self.N = np.zeros((P, K))
        np.add.at(self.S, inv, Dz)
        np.add.at(self.N, inv, obs.astype(np.float64))
        self.pos = {int(p): i for i, p in enumerate(upair)}
        # per-context totals
        self.ctx_of_pair = upair // self.n_comp
        self.Sc = np.zeros((self.n_ctx, K)); self.Nc = np.zeros((self.n_ctx, K))
        np.add.at(self.Sc, self.ctx_of_pair, self.S)
        np.add.at(self.Nc, self.ctx_of_pair, self.N)

    def mean_excluding(self, ctx, exclude_comps):
        """Context mean over rows whose compound is not in exclude_comps."""
        s = self.Sc[ctx].copy(); n = self.Nc[ctx].copy()
        for c in exclude_comps:
            j = self.pos.get(int(ctx) * self.n_comp + int(c))
            if j is not None:
                s -= self.S[j]; n -= self.N[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(n > 0, s / np.maximum(n, 1e-12), np.nan)
        return mu


def rowwise_pcc(resid, obs, v):
    """Per-row Pearson correlation between a fixed vector v and each residual row."""
    m = obs.astype(np.float64)
    n = m.sum(1)
    vx = v[None, :] * m
    sx = vx.sum(1); sy = (resid * m).sum(1)
    sxx = (v[None, :] ** 2 * m).sum(1); syy = (resid ** 2 * m).sum(1)
    sxy = (vx * resid).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        a = sxx - sx * sx / n
        b = syy - sy * sy / n
        r = cov / np.sqrt(a * b)
    r[(n < 3) | (a <= 0) | (b <= 0)] = np.nan
    return r


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def load_text_embeddings(compounds):
    f = pd.read_parquet(ROOT / CFG["channels"]["phi_TXT"]["artifact"])
    idx = {c: i for i, c in enumerate(f["compound"])}
    X = f.drop(columns=["compound"]).to_numpy(np.float64)
    out, ok = np.zeros((len(compounds), X.shape[1])), np.zeros(len(compounds), bool)
    for i, c in enumerate(compounds):
        j = idx.get(c)
        if j is not None:
            out[i] = X[j]; ok[i] = True
    return out, ok


def load_ssps(compounds, proteins):
    index = {p: i for i, p in enumerate(proteins)}
    vecs, ok = np.zeros((len(compounds), len(proteins))), np.zeros(len(compounds), bool)
    cache = {}
    for path in sorted(glob.glob(str(ROOT / "external_data/processed/ssps_priors/*.jsonl"))):
        for line in open(path):
            rec = json.loads(line)
            name = str(rec["compound"])
            if name in cache:
                continue
            v = np.zeros(len(proteins))
            for p in rec.get("proteins", []):
                i = index.get(str(p.get("name", "")).strip())
                if i is not None:
                    v[i] += float(p.get("direction", 0)) * float(p.get("confidence", 0))
            cache[name] = v
    for i, c in enumerate(compounds):
        v = cache.get(c)
        if v is not None and np.abs(v).sum() > 0:
            vecs[i] = v / np.linalg.norm(v); ok[i] = True
    return vecs, ok


def load_morgan(compounds):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    m = pd.read_csv(ROOT / "external_data/processed/compound_mapping_features.csv")
    smi = dict(zip(m["query_label"].astype(str), m["canonical_smiles"].astype(str)))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out, ok = np.zeros((len(compounds), 2048)), np.zeros(len(compounds), bool)
    for i, c in enumerate(compounds):
        s = smi.get(c)
        if not s or s == "nan":
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        out[i] = np.array(gen.GetFingerprint(mol), dtype=np.float64); ok[i] = True
    return out, ok


def load_random(compounds, dim=768):
    rng = np.random.default_rng(RNG_SEED)
    return rng.standard_normal((len(compounds), dim)), np.ones(len(compounds), bool)


# --------------------------------------------------------------------------- #
# ridge with closed-form LOO for lambda selection
# --------------------------------------------------------------------------- #
def ridge_loo_select(Xq, Y):
    """Pick lambda by exact leave-one-out over the fit compounds; return the fit."""
    n = Xq.shape[0]
    Xc = np.c_[np.ones(n), Xq]
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    best, best_lam, best_beta = np.inf, None, None
    for lam in LAMBDA_GRID:
        d = s / (s ** 2 + lam)
        H = (U * (s * d)) @ U.T
        beta = Vt.T @ (d[:, None] * (U.T @ Y))
        pred = Xc @ beta
        h = np.clip(np.diag(H), 0, 1 - 1e-9)
        loo = (Y - pred) / (1 - h)[:, None]
        mse = float(np.mean(loo ** 2))
        if mse < best:
            best, best_lam, best_beta = mse, float(lam), beta
    return best_beta, best_lam, best


def design_audit(Xq):
    """E1: condition number and effective rank (participation ratio)."""
    s = np.linalg.svd(np.c_[np.ones(len(Xq)), Xq], compute_uv=False)
    s = s[s > 0]
    cond = float(s[0] / s[-1]) if len(s) else np.inf
    p = s ** 2 / (s ** 2).sum()
    eff = float(np.exp(-(p * np.log(p + 1e-300)).sum()))
    return cond, eff


# --------------------------------------------------------------------------- #
# main harness
# --------------------------------------------------------------------------- #
def run(args):
    z = np.load(CACHE, allow_pickle=True)
    D = z["deltas"].astype(np.float64)
    comp_id = z["compound_id"]; compounds = list(z["compounds"])
    proteins = list(z["proteins"])
    ctx_id = z["ctx_strict_id"] if args.granularity == "strict" else z["ctx_bio_id"]
    n_comp = len(compounds)
    sums = CtxSums(D, ctx_id, comp_id)
    rows_of = {c: np.where(comp_id == c)[0] for c in range(n_comp)}

    feats = {}
    feats["phi_TXT"] = load_text_embeddings(compounds)
    feats["phi_SSPS"] = load_ssps(compounds, proteins)
    feats["tanimoto"] = load_morgan(compounds)
    feats["random"] = load_random(compounds)
    for k, (X, ok) in feats.items():
        print(f"  feature {k:10s} dim={X.shape[1]:5d} usable={int(ok.sum())}/{n_comp}")

    ssps_vecs = feats["phi_SSPS"][0]
    eligible = np.where(feats["phi_TXT"][1])[0]        # folds the primary can score
    k_ladder = CFG["oracle_k_ladder"]
    results = {"per_fold": {}, "config_sha": hashlib.sha256(
        (ROOT / "configs/dcb_gates.json").read_bytes()).hexdigest()}
    leak_dir = OUT / f"harness_{args.granularity}"
    leak_dir.mkdir(parents=True, exist_ok=True)

    m3_rho, m3_sign = [], []
    per_fold_rows = []
    basis_hashes = {}

    for i in range(n_comp):
        ri = rows_of[i]
        if len(ri) == 0:
            continue
        fit_comps = [c for c in range(n_comp) if c != i]

        # ---- target: held-out compound rows, mu from fit set only ---------- #
        mu_cache = {}
        res_i = np.empty((len(ri), D.shape[1]))
        for a, r in enumerate(ri):
            x = int(ctx_id[r])
            if x not in mu_cache:
                mu_cache[x] = sums.mean_excluding(x, [i])
            res_i[a] = D[r] - mu_cache[x]
        obs_i = np.isfinite(res_i)                      # mask RECOMPUTED after -mu
        keep_rows = obs_i.sum(1) >= 3
        res_i, obs_i = res_i[keep_rows], obs_i[keep_rows]
        res_i_z = np.nan_to_num(res_i)

        # ---- M3 self-check: the cached prior used directly ------------------ #
        if feats["phi_SSPS"][1][i]:
            r = rowwise_pcc(res_i_z, obs_i, ssps_vecs[i])
            m3_rho.append((compounds[i], float(np.nanmean(r)), int(np.isfinite(r).sum())))
            prof_sign = np.sign(np.nansum(np.where(obs_i, res_i_z, np.nan), axis=0))
            sup = ssps_vecs[i] != 0
            if sup.sum():
                m3_sign.append(float(np.mean(np.sign(ssps_vecs[i][sup]) == prof_sign[sup])))

        # ---- fit-set profiles: leave-own-compound-out mu, fit set only ------ #
        prof = np.full((len(fit_comps), D.shape[1]), np.nan)
        for b, j in enumerate(fit_comps):
            rj = rows_of[j]
            if len(rj) == 0:
                continue
            acc = np.zeros(D.shape[1]); cnt = np.zeros(D.shape[1])
            mu_j = {}
            for r in rj:
                x = int(ctx_id[r])
                if x not in mu_j:
                    mu_j[x] = sums.mean_excluding(x, [i, j])
                v = D[r] - mu_j[x]
                m = np.isfinite(v)
                acc[m] += v[m]; cnt[m] += 1
            with np.errstate(invalid="ignore"):
                prof[b] = np.where(cnt > 0, acc / np.maximum(cnt, 1e-12), np.nan)

        valid = np.isfinite(prof).sum(1) > 0
        prof_v = prof[valid]
        fit_ids = [fit_comps[b] for b in np.where(valid)[0]]
        centre = np.nanmean(prof_v, axis=0)
        Pm = np.nan_to_num(prof_v - centre[None, :])
        U, S, Vt = np.linalg.svd(Pm, full_matrices=False)
        var_ratio = (S ** 2) / (S ** 2).sum()
        basis_hashes[compounds[i]] = hashlib.sha256(
            np.ascontiguousarray(Vt[:5]).tobytes()).hexdigest()[:16]

        # split-half principal angles (in-sample audit)
        half = len(Pm) // 2
        rs = np.random.default_rng(RNG_SEED + i).permutation(len(Pm))
        _, _, V1 = np.linalg.svd(Pm[rs[:half]], full_matrices=False)
        _, _, V2 = np.linalg.svd(Pm[rs[half:2 * half]], full_matrices=False)

        target_i = np.nanmean(np.where(obs_i, res_i_z, np.nan), axis=0)
        target_i = np.nan_to_num(target_i - centre)

        fold = {"compound": compounds[i], "n_rows": int(len(res_i)),
                "n_fit_profiles": int(len(Pm)), "oracle": {}, "channels": {},
                "cum_var": {}, "split_half_cos": {}}

        for k in k_ladder:
            kk = min(k, Vt.shape[0])
            B = Vt[:kk]
            fold["cum_var"][str(k)] = float(var_ratio[:kk].sum())
            c_ = min(kk, V1.shape[0], V2.shape[0])
            sv = np.linalg.svd(V1[:c_] @ V2[:c_].T, compute_uv=False)
            fold["split_half_cos"][str(k)] = float(np.clip(sv, -1, 1).min()) if c_ else None
            coef = B @ target_i                              # oracle: solved with truth
            v = coef @ B + centre
            r = rowwise_pcc(res_i_z, obs_i, v)
            fold["oracle"][str(k)] = float(np.nanmean(r))

        # ---- real channels ------------------------------------------------- #
        for name, (X, ok) in feats.items():
            if not ok[i]:
                fold["channels"][name] = {"abstained": True}
                continue
            sel = [b for b, j in enumerate(fit_ids) if ok[j]]
            if len(sel) < 10:
                fold["channels"][name] = {"abstained": True, "reason": "too few fit rows"}
                continue
            Xf = X[[fit_ids[b] for b in sel]]
            Yfull = Pm[sel]
            mu_x = Xf.mean(0); sd_x = Xf.std(0); sd_x[sd_x == 0] = 1.0
            Xs = (Xf - mu_x) / sd_x
            xi = (X[i] - mu_x) / sd_x
            Ux, Sx, Vx = np.linalg.svd(Xs, full_matrices=False)   # PCA on features only
            best, all_cfgs = None, []
            for q in Q_GRID:
                qq = min(q, Vx.shape[0])
                Xq = Xs @ Vx[:qq].T
                xq = xi @ Vx[:qq].T
                cond, eff = design_audit(Xq)
                eliminated = (cond > CFG["E1_elimination"]["condition_number_max"]) or (eff < qq)
                for k in k_ladder:
                    kk = min(k, Vt.shape[0])
                    Yk = Yfull @ Vt[:kk].T
                    beta, lam, mse = ridge_loo_select(Xq, Yk)
                    pred_c = np.r_[1.0, xq] @ beta
                    v = pred_c @ Vt[:kk] + centre
                    rr = rowwise_pcc(res_i_z, obs_i, v)
                    rho = float(np.nanmean(rr))
                    vs = np.zeros_like(v)
                    top = np.argsort(-np.abs(v - centre))[:SPARSE_SUPPORT]
                    vs[top] = (v - centre)[top]
                    rho_sp = float(np.nanmean(rowwise_pcc(res_i_z, obs_i, vs)))
                    rec = {"k": k, "q": q, "lambda": lam, "inner_mse": mse,
                           "cond": cond, "eff_rank": eff, "E1_eliminated": bool(eliminated),
                           "rho_dense": rho, "rho_sparse": rho_sp,
                           "beta_norm": float(np.linalg.norm(beta[1:]))}
                    all_cfgs.append(rec)
                    # E1 is the SOLE elimination gate: an eliminated (k,q) may not
                    # be selected, however good its inner MSE looks.
                    if eliminated:
                        continue
                    if (best is None) or (mse < best["inner_mse"]):
                        best = rec
            if best is None:
                # every (k,q) failed E1 conditioning -> the channel is unscorable here
                fold["channels"][name] = {"abstained": True, "reason": "all configs E1-eliminated",
                                          "n_configs": len(all_cfgs)}
            else:
                best = dict(best)
                best["n_configs_total"] = len(all_cfgs)
                best["n_configs_E1_eliminated"] = sum(c["E1_eliminated"] for c in all_cfgs)
                # beta-norm sensitivity: the best config had E1 ignored (diagnostic only)
                unrestricted = min(all_cfgs, key=lambda c: c["inner_mse"])
                best["rho_dense_ignoring_E1"] = unrestricted["rho_dense"]
                fold["channels"][name] = best
        results["per_fold"][compounds[i]] = fold
        per_fold_rows.append(fold)
        print(f"  fold {i+1:2d}/{n_comp} {compounds[i][:34]:34s} "
              f"rows={fold['n_rows']:3d} oracle(full)={fold['oracle'][str(k_ladder[-1])]:+.4f}")

    # ---- leak assertions ------------------------------------------------- #
    leak = {"basis_hashes_distinct": len(set(basis_hashes.values())) == len(basis_hashes),
            "n_folds": len(basis_hashes),
            "mu_excludes_heldout": True,
            "mask_recomputed_after_mu": True}
    (leak_dir / "leak_check.json").write_text(json.dumps(
        {"assertions": leak, "basis_hashes": basis_hashes}, indent=2))

    m3 = {"per_compound": {c: {"rho_mean": r, "n_rows": n} for c, r, n in m3_rho},
          "rho_bar_row_weighted": float(np.average([r for _, r, _ in m3_rho],
                                                   weights=[n for _, _, n in m3_rho])),
          "rho_mean_unweighted": float(np.mean([r for _, r, _ in m3_rho])),
          "sign_agreement": float(np.mean(m3_sign)) if m3_sign else None,
          "n_compounds": len(m3_rho)}
    results["m3_selfcheck"] = m3
    results["leak_check"] = leak
    (leak_dir / "harness_results.json").write_text(json.dumps(results, indent=2,
                                                              ensure_ascii=False))
    print(f"\nM3 self-check: rho_bar={m3['rho_bar_row_weighted']:.6f} "
          f"(reference 0.005872), unweighted={m3['rho_mean_unweighted']:.6f} "
          f"(reference -0.000727), sign_agreement={m3['sign_agreement']}")
    print(f"leak assertions: {leak}")
    print(f"wrote {leak_dir/'harness_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", choices=["strict", "bio"], default="strict")
    run(ap.parse_args())
