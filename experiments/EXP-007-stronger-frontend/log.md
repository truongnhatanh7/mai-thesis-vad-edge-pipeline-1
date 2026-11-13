# EXP-007 log (append-only)

### 2026-11-06 20:29 — Lead: plan frozen, weights, caches launched
- `plan.md` frozen 20:45 before any run (D19). YOLOv8s-pose from Ultralytics assets v8.4.0, 23,513,657 bytes,
  sha256 `234314cd8baf62616791aceb9ea6ad5c19f26cf6c0d8f3a1bfce1e23b186cfb3`, at `artifacts/EXP-001/models/`.
- 9 caches in tmux `exp007` (`~/thesis-data/exp007-sht/sweep.sh`): ShanghaiTech L0 E0, L2 {E0,E2}, L3 {E0,E1};
  UBnormal L0 E0, L3 {E0,E1,E2}; then gate_sweep per dataset (MLflow `EXP-007`). ETA: ShanghaiTech ≈ 2 h,
  UBnormal dark ≈ 6–7 h (CPU-bound darkening of 1280×720 frames).

### 2026-11-07 07:17 — Lead: caches done (20:54–02:40), scored 03:22; analysis + results.md
- V1 fail, V2 half, V3 fail (gamma), V4 pass. Numbers in results.md (inline analysis with exp003/exp005 functions).
