# Security

wormOCR is a local OCR library. It does not send images or text to a service.

## Checkpoints

`.pt` files are PyTorch pickle archives. Loading one can execute code embedded in the file. Only load checkpoints you created or received from someone you trust. Do not point `--checkpoint` at a file from an untrusted download.

ONNX files are a better artifact to pass around when you only need inference, because the runtime does not unpickle arbitrary Python. Verify the numeric parity printed by `wormocr export` after you create one.

## Reports

If you believe a checkpoint, the CLI, or a parser can be abused in a way that puts users at risk, open a GitHub security advisory on [arjun988/wormOCR](https://github.com/arjun988/wormOCR) or contact the maintainer listed in the license. Please include a minimal file or command that reproduces the issue, and avoid posting an exploit on a public issue if it is still unfixed.

There is no bug bounty.
