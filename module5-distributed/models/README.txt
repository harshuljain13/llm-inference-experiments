For Docker Compose (CPU profile), put your weights in this folder as:

  model.gguf

Example:
  cp ~/Downloads/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf ./model.gguf

Or mount another directory: MODEL_DIR=/path/to/dir (must contain the file named in LLAMA_MODEL_PATH, default /models/model.gguf inside the container).
