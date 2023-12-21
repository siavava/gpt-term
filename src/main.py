#!/usr/bin/env python3
from network import GPTLanguageModel, load_model, torch, device, TextProcessor

def main():
  model: GPTLanguageModel = load_model("checkpoints/transfusion.pth")
  response = None
  while query := input("QUERY? "):
    if query == "-": query = response or "Empty query"
    context = torch.tensor( [TextProcessor.encode(query)], dtype=torch.long, device=device)
    response = TextProcessor.decode(model.generate(context, count=150)[0].tolist())
    print(f"\nRESPONSE:\n{response}\n\n")

if __name__ == "__main__":
  main()
