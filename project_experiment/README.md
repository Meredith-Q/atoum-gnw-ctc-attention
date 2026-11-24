# CTC experiment workspace

Stage 0 through Stage 3 call the original 40-area `model_class.Model` dynamics via `original_model_runner.py`; the optional phase coupling and E→E gate are added by `ctc_model.CTCModel`. Stage 3 has been migrated, but its six scans have not yet been rerun with this implementation.

Stage 2 now targets visual attention: V4 receives the visual stimulus, a provisional frontal node (8l) receives the attention cue, and phase interactions/gating are restricted to 8l, V1, V2, and V4. It compares directed FLN phase coupling, common rhythm, and rate-only controls. The previous joint-threshold notebook is now Stage 3. Stage 2 pilot execution is opt-in and initially uses four paired seeds.

The old standalone `gnw_mean_field.py` implementation was removed. Previous pilot CSV and PNG outputs are preserved under `legacy_two_node_prototype_results/` for provenance only. They do not validate the migrated original-model runs. Run the notebooks again to generate current results in this folder.




## Stage 0 current candidate

The streamlined `CTC Stage 0 - Original Model.ipynb` uses the original population-rate model with the explicit phase-gate extension. Active settings: V4 stimulus 500 pA for 1000 ms, gate depth 1.0, K=10.8 rad/s. An expanded 24-seed check found 2/24 ignition (8.3%; seeds 12 and 23; Wilson 95% CI 2.3%–25.8%). Exploratory pilots varied stimulus strength (400–800 pA), duration (500–1500 ms), K (4–14 rad/s), and gate depth (0–1); none improved beyond 1/3 in the three-seed screening set. The current candidate is therefore not a robust ignition regime. A matched-seed scan of the existing FLN long-range E→E scale (`g_sw`, 0–3× native) now covers 12 phase seeds (192 trials across gate states and gains). The FLN pathway transmits rate activity, and phase gating raises workspace rates near native gain, but sustained ignition was only 1/12 at 1.0–1.5× with the gate on and 0/12 otherwise. Highest observed single-area rate was about 51 Hz. Results are in `stage0_fln_gain_expanded_trials.csv`, `stage0_fln_gain_expanded_summary.csv`, and `stage0_fln_gain_expanded.png`.

The optional `workspace_reentry_current_pA` extension was then compared at 0 versus 50 pA over 24 paired seeds. Ignition rose from 2/24 to 24/24 and post-stimulus sustained ignition from 0/24 to 24/24; max single-area rate rose from about 49 to 52 Hz. This is an explicit positive-feedback extension that maintains recruited workspace activity, not a result of the unmodified model or proof of CTC. Results are in `stage0_workspace_reentry_confirm_*`.

## Stage 1: Stage 0-matched A/B/C comparison

The Stage 1 24-seed re-entry-factor comparison found that at 0 pA workspace re-entry, A ignited 2/24 and C 18/24; at 50 pA, A ignited 24/24 and C 23/24. This supports activity maintenance by the added feedback but does not show an advantage of reciprocal CTC over the one-way control. B stays near zero because long-range E→E transmission is disabled. Results are in `stage1_ctc_reentry_factor_24seed_*`; the earlier Stage 1 pilot files remain separate.

The Stage 1 MI circular-shift null uses 10 surrogates in this pilot. Increase paired seeds and surrogate count only after Stage 0 yields a reliable, non-ceiling candidate.


## Stage 3 status

`CTC Stage 3 - Joint Threshold Analysis.ipynb` contains six opt-in scans using the original model. Prototype-era pilot summaries were removed from the notebook; their data remain archived under `legacy_two_node_prototype_results/`. Stage 3 output files use the `stage3_*` prefix and will be generated when the corresponding scan cells are run. The Stage 1 candidate K and stimulus settings are exploratory inputs, not a confirmed CTC parameter window.
