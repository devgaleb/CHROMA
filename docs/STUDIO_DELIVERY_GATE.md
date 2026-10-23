# CHROMA Studio Delivery Gate

This release adds integrity verification and explicit production approval gates to CHROMA shot archives. It **does not** certify renders as studio-grade by itself.

## Use

```bash
python chroma_delivery.py renders/SH010 --shot SH010 --start 1001 --end 1240 --output SH010_v001.zip
python chroma_studio_gate.py SH010_v001.zip --spec studio_requirements.json --report gate_result.json
```

Sample `studio_requirements.json`:

```json
{
  "expected_frames": 240,
  "require_color_evidence": true,
  "color_pipeline": {
    "working_space": "ACEScg",
    "display_space": "Rec.709",
    "ocio_config_sha256": "REPLACE_WITH_REAL_OCIO_CONFIG_CHECKSUM"
  },
  "required_approvals": ["vfx_supervisor", "color_supervisor"],
  "approvals": {},
  "quality_checks": {
    "alpha_mae": {"maximum": 0.02, "required": true},
    "temporal_flicker": {"maximum": 0.01, "required": true}
  }
}
```

Populate measured values only from actual validation runs; approvals must come from authorized reviewers. The gate checks **presence of OCIO metadata**, not numerical color accuracy. A calibrated color-validation suite, trained neural weights, actual 4K/8K render tests and multiple-GPU farm trials are separate requirements. Do not claim them complete based on this gate.
