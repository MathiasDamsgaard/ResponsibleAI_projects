from llama_cpp import Llama

# command to download specific model
# huggingface-cli download TheBloke/openchat-3.5-0106-GGUF openchat-3.5-0106.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
# credit to https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF


def init_model(model_path="./models/openchat-3.5-0106.Q4_K_M.gguf"):
    return Llama(
        model_path=model_path,  # Download the model file first
        n_ctx=150,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=35,  # The number of layers to offload to GPU, if you have GPU acceleration available
        verbose=False,  # Whether to print out detailed logs
    )


def get_response(prompt, model, temperature=1.2):
    output = model(
        f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:",  # Prompt
        max_tokens=30,  # Generate up to 512 tokens
        stop=[
            "</s>"
        ],  # Example stop token - not necessarily correct for this specific model! Please check before using.
        echo=False,  # Whether to echo the prompt
        temperature=0.9,  # The temperature to use DEFAULT 0.8
        top_p=0.95,  # The nucleus sampling probability DEFAULT 0.95
        # top_k=40,  # The top k tokens to use for top-k sampling DEFAULT 40
    )
    return output["choices"][0]["text"]

