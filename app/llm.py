import dspy


def get_llm():
    llm_model = dspy.LM(
        "ollama_chat/gpt-oss:20b", api_base="http://localhost:11434", api_key=""
    )
    dspy.configure(lm=llm_model)
    return llm_model
