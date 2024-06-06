# VIVset Benchmark

## Installation

To install our benchmark please clone this repository to your workspace.

To properly use all of our code you have to have Python 3 and pip installed.

Next up you should install all required packages through a terminal in your workspace directory:

```bash
# Python 3.10+
pip install -r requirements.txt
```

To use any of the code you should also download the dataset that is available <a href="https://drive.google.com/drive/folders/1VNG6wt47lOu4LjDK3w9OHWqE7qkRPPHb">here</a>.

The contents of the dataset should be copied into the 'environment' folder of this repository.

## Walkthrough of our Repository

Our repository should be used to run our benchmark pipeline with different configurations.

### Running the Benchmark Pipeline

Secondly, we have implemented our benchmark pipeline in this repository. It can be used to verify our measured outcomes though it has to be noted that Large multi-modal models (LMM)s tend to show indeterministic behaviour eventhough parameters are set accordingly. To run the benchmark use the following steps.

First of all, set your API keys for the LMM that you chose as an environment variable:

Use the environment variable 'OPENAI_API_KEY' for using GPT-4-Vision-Preview.

Use the environment variable 'ANTHROPIC_API_KEY' for using Claude 3 Opus.

Use the environment variable 'GEMINI_API_KEY' for using Gemini-1.5-Pro.

Next up you should run the script 'run_pipeline.py'. If you run the script it will load the sequences for the specified room and precaution available in the environment and use baseline0 and GPT-4-Vision-Preview to interpret the sequence. If you want to change the parameters pass the following arguments:

1.: Baseline (baseline0, baseline1, baseline2)

2.: LMM client type (openai, gemini, claude)

3.: Rooms (surgery, icu, labour)
    Rooms should be passed as a list: e.g. 'surgery' 'icu'

4.: Precautions (contact, contact_plus, droplet, neutropenia, covid)
    Precautions should be passed as a list: e.g. 'contact' 'contact_plus'
    The Special Respiratory Precautions (covid) are only available in the surgery room

5.: Number of Iterations (integer greater than 0)

A typical call could look something like this:

```bash
python run_pipeline.py --baseline "baseline0" --llm "openai" --rooms "surgery" "icu" --precautions "contact" "droplet" --iterations 2
```

Results of the algorithm are saved as CSV files in the 'test_sequences/PRECAUTION/results' folders in the environment.