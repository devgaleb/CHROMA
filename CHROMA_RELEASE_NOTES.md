# CHROMA production upgrade

This release adds a Gradio temporal AI render tab producing 16-bit PNG alpha sequences, checkpoint and video validation, recurrent state controls, and a render farm fix for atomic PNG output.

**Neural prerequisites:** Install a compatible licensed PyTorch model package exposing `model.MattingNetwork`, plus PyTorch, and supply legitimate `.pth` weights. No newly trained weights are claimed or bundled. Verify licensing before distributing model code.

**Not production certified:** No complete 4K/8K render, distributed multi-GPU soak test, model accuracy validation, or ACES compliance test has been run.
