import torch
from inference.generate import MAX_NEW_TOKENS, load_checkpoint, select_model
from model import config
from model.tokenizers import sentence_piece_tokenizer


def main() :

    checkpoint_dir = select_model()
    model, tokenizer = load_checkpoint(checkpoint_dir)

    conversation = ""

    while True :
        user = input("Your Message: ")
        conversation += user

        tokens = tokenizer.encode(conversation)

        context = torch.tensor([tokens], device = config.device)

        output = model.generate(context, max_new_tokens=100)

if __name__ == "__main__":
    main()
