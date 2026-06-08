from src.finetune.model import LLM_Model


def main():
    print("Loading model...")
    llm = LLM_Model()
    info = llm.get_model_and_processor()
    print(f"Model:  {info['model_name']}")
    print(f"Device: {info['device']}")
    print("Model loaded successfully!")
    print("Generating response for prompt: 'What is the capital of France?'")
    response = llm.generate_response("What is the capital of France?")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
