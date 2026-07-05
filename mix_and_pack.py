"""Knapsack-mix multiple rate runs per image under avg-bpp budget, pack submission.zip.

Usage: python mix_and_pack.py m_ft8.json m_ft4.json m_ft2.json
(metrics jsons downloaded from volume; order irrelevant)
Outputs: plan.json (per-image chosen run) + submission builder commands.
"""
import json, sys, itertools, pathlib

BUDGET = 0.025
LP = "lpips_alex"  # assumed organizer backbone; vgg tracked too


def score(r, lp=LP):
    return r["psnr"] + 10 * r["msssim"] + 40 * (1 - r[lp]) + 40 * (1 - r["dists"])


def main(files):
    runs = {}
    for f in files:
        d = json.load(open(f))
        tag = pathlib.Path(f).stem[2:]  # m_<run_dir_name>.json
        runs[tag] = {r["name"]: r for r in d["per_image"]}
    names = sorted(next(iter(runs.values())))
    tags = list(runs)

    # start: everyone on cheapest run; greedy upgrades by marginal score/bit
    choice = {}
    for n in names:
        choice[n] = min(tags, key=lambda t: runs[t][n]["bpp"])

    def avg_bpp():
        return sum(runs[choice[n]][n]["bpp"] for n in names) / len(names)

    improved = True
    while improved:
        improved = False
        best = None
        for n in names:
            cur = runs[choice[n]][n]
            for t in tags:
                cand = runs[t][n]
                dbits = cand["bpp"] - cur["bpp"]
                dscore = score(cand) - score(cur)
                if dscore > 1e-6 and dbits > 0:
                    eff = dscore / dbits
                    if best is None or eff > best[0]:
                        best = (eff, n, t, dbits)
                elif dscore > 1e-6 and dbits <= 0:  # free win
                    choice[n] = t
                    improved = True
        if best and not improved:
            _, n, t, dbits = best
            old = choice[n]
            choice[n] = t
            if avg_bpp() > BUDGET:
                choice[n] = old
            else:
                improved = True

    sel = [runs[choice[n]][n] for n in names]
    mean = lambda k: sum(r[k] for r in sel) / len(sel)
    agg = {k: mean(k) for k in ("bpp", "psnr", "msssim", "lpips_alex", "lpips_vgg", "dists")}
    agg["score_alex"] = sum(score(r, "lpips_alex") for r in sel) / len(sel)
    agg["score_vgg"] = sum(score(r, "lpips_vgg") for r in sel) / len(sel)
    plan = {n: choice[n] for n in names}
    json.dump({"plan": plan, "mean": agg}, open("plan.json", "w"), indent=1)
    print(json.dumps(agg, indent=1))
    from collections import Counter
    print(Counter(plan.values()))


if __name__ == "__main__":
    main(sys.argv[1:])
