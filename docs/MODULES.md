# Video training modules

The module names follow the method section of *Multi-modal Video Fusion under Frame-Rate Mismatch*.

| Module | Implementation | Role |
| --- | --- | --- |
| CWTM — Confidence-Weighted Temporal Modeling | `asyncfusion/archs/cwtm.py` | Temporal feature modeling weighted by timestamp-derived infrared confidence. |
| VGEA — Visible-Guided Evidence Aggregation | `asyncfusion/archs/vgea.py` | Confidence-gated visible evidence aggregation for stale infrared features, with motion-based feature transport. |
| TCR — Temporal Context Refinement | `asyncfusion/archs/tcr.py` | Temporal attention and refinement at the decoder scales. |

`asyncfusion/archs/asyncfusion_arch.py` assembles the three modules into a multi-scale fusion network. Public module names are independent of checkpoint parameter keys.

## Training objective

`asyncfusion/models/async_fusion_model.py` combines fusion quality, modality reconstruction fidelity, temporal consistency, a motion-weighted loss, and distillation from a frozen synchronous teacher. Synchronous target clips are loaded at the original query timestamps; asynchronous infrared input clips follow `schedule.json`.

The implementation computes motion weights from absolute temporal differences in the dense visible video. The teacher uses the same architecture and receives synchronized clips without lag conditioning. The teacher checkpoint must be supplied locally and remains frozen throughout training.
