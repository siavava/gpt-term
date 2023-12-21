# Generative Pre-trained Transformer Model

This is an implementation of the GPT model from the paper [Language Models are Unsupervised Multitask Learners][gpt],
which is itself modelled after the [transformer][transformer] neural network architecture.

## Usage

### 1. Neural Network

[`network.py`](src/network.py) contains the implementation of the GPT model.
  It exports the following:
  - `GPTLanguageModel` class: The GPT model.
  - `loadmodel` function: loads the latest training checkpoint
    (see the [checkpoints directory](checkpoints)).

> [!WARNING]
>
> To trim the codebase, I removed all unneeded code
> including the training routine.
> If interested in seeing that, check out the [original repository][transfusion].

### 2. Sample Driver Code

[`main.py`](src/main.py) contains a simple CLI program
  that loads the latest checkpoint and generates text from it.

To use it, make sure you install the dependencies first:

```bash
# using pip
pip install -r requirements.txt

# or... using conda
conda env create -f environment.yml
```

Then, run the program:

```bash
python3 src/main.py
```

- Once the program launches, you will be prompted to enter a prompt.
  The model will then generate text based on that prompt.  
- _You can also type in `-` to use the previous generation
  as the current prompt, effectively chaining generations._
- An empty query will terminate the program.

#### Sample Run

```text
/Users/amittaijoel/workspace/               
λ> python3 src/main.py
QUERY? OpenAI is going to

RESPONSE:
OpenAI is going to excause AI, but I also the
future. What if we just have anyone electulessia, it may
actually said human-relassively if we are these investigating whe


QUERY? -

RESPONSE:
OpenAI is going to excause AI, but I also the
future. What if we just have anyone electulessia, it may
actually said human-relassively if we are these investigating where are
dealizing is broad from an emerging about data statement," really
acreed investigations with their own Afailing to generate competition. Busine


QUERY? -       

RESPONSE:
OpenAI is going to excause AI, but I also the
future. What if we just have anyone electulessia, it may
actually said human-relassively if we are these investigating where are
dealizing is broad from an emerging about data statement," really
acreed investigations with their own Afailing to generate competition. Business
increase, he found in was how to the maindernationalist didn’t,
it wanted after them because you need to believe boosts
with their human within col


QUERY? 
λ> 
```


[gpt]: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
[transformer]: https://arxiv.org/abs/1706.03762
[transfusion]: https://github.com/empirical-studio/transfusion
