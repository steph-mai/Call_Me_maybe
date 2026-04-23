*This project has been created as part of the 42 curriculum by stmaire.*

<div align="center">
<br>
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTQPzuYKu7n0cWUYa5Kbg0_LrlEQAIURWeo9A&s" alt="42 Logo" width="400" />

  <br>
</div>

# Call Me Maybe - LLM Function Calling

![Language](https://img.shields.io/badge/Language-python-blue)
![Static Badge](https://img.shields.io/badge/AI-pink)
![Static Badge](https://img.shields.io/badge/parsing-pink)
![Static Badge](https://img.shields.io/badge/LLM-pink)

![Tag](https://img.shields.io/badge/Unit_tests-green)
## 🔵 Description

### ✳️ Goal

Create a function calling tool that translates natural language
prompts into structured function calls. 

### ✳️ Overview

The project is structured as a modular **pipeline** that handles everything from raw text input to validated JSON output. It is divided into four main logic layers:

### 1. Input Parsing & Validation Layer (models.py & load.py)
To ensure the pipeline operates on reliable data, a dedicated parsing layer was implemented. The input files data are validated using **Pydantic objects**.

### 2. Constraint Layer (ConstrainedDecoder & StateNode):
This is the technical core of the project. It uses a **Trie (Prefix Tree)** to manage function names and **regex-based filters** for parameters. By intercepting the model's logits, it mathematically prevents the generation of invalid tokens.

### 3. Orchestration Layer (PromptProcessor):
This component manages the execution flow. It formats the system prompts, handles iterative parameter extraction (looping through each required argument), and implements a caching system to optimize performance by skipping redundant LLM calls.

### 4. Data & Validation Layer (models.py & Recovery):
Using Pydantic models after generation, an "Advanced Recovery" system performs final "surgical" cleaning to ensure the final JSON file is production-ready.


## 🔵 Instructions

### ✳️ Prerequisites

- Python 3.10 or later.

- uv (high-performance Python package manager).

### ✳️ Installation

The project uses uv for dependency management and isolation. To set up the environment and install necessary packages (numpy, pydantic, flake8, mypy, pytest):

```Bash
# Install dependencies and create virtual environment
uv sync
```
Using the Makefile: 
```bash
make install
```
### ✳️ Execution

* ### Using the Makefile (Recommended)
```Bash
# Run with default paths (data/input/ and data/output/)
make run
```
Clean cache and temporary files (__pycache__, .mypy_cache, etc.)

```
make clean
```
* ### Manual Execution via uv
**1. Default execution**
```Bash
uv run python -m src
```

**2. Execution with custom arguments**
```Bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```
**3. Testing with a different model (e.g., SmolLM2)**
```Bash
 uv run python -m src --model HuggingFaceTB/SmolLM2-360M-Instruct
```
Note: While the project is optimized for Qwen3-0.6B, the constrained decoding logic has been validated with SmolLM2-135M to ensure robustness across different tokenizers.

**4. Development & Quality Control**
In accordance with the 42 curriculum standards, the project adheres to strict coding rules:

Linting: Both flake8 and mypy are used to ensure PEP 8 compliance and strict type safety.

```Bash
make lint        # Standard check
make lint-strict # Enhanced strict checking
```

**5. Debugging**
To run the script in debug mode with pdb:

```Bash
make debug
```

**6. Testing**
While not graded, a suite of tests is included to verify the constrained decoding logic:

```Bash
make test
```


## 🔵 Resources

### ✳️ References
To build this project, several key concepts were explored:

* Logit Masking & Constrained Decoding: https://huggingface.co/docs

* Trie or prefix tree: https://fr.wikipedia.org/wiki/Trie_(informatique)

* State Machine: https://fr.wikipedia.org/wiki/Automate_fini

* LLM et SLM : 
https://www.ibm.com/fr-fr/think/topics/small-language-models
https://blog.stephane-robert.info/docs/developper/programmation/python/llm/

* Qwen3-0.6B: https://qwen.ai/blog?id=qwen3

* argparse: tutoriel: https://docs.python.org/fr/3/howto/argparse.html

* uv: https://www.datacamp.com/fr/tutorial/python-uv?dc_referrer=https%3A%2F%2Fwww.google.com%2F

* numpy : https://numpy.org/doc/stable/

### ✳️ AI Usage
Strategic Planning: AI was used at the beginning of the project to brainstorm the modular decomposition of the pipeline (Separation of ConstrainedDecoder vs PromptProcessor).

Assistance in writing complex Regular Expressions for the logit masks.

Help with debugging specific numpy slicing operations and mypy type-hinting errors.

Testing: AI helped in creating the pytest structure and identifying edge cases for the function_calling_edge_cases.json file.

Documentation: Assistance in translating this README.md.


## 🔵 Algorithm explanation

### Constrained Decoding ###

The core of this project lies in Constrained Decoding, a technique used to guide the LLM's generation process. 
Instead of letting the model choose from its entire vocabulary, we restrict its choices at each step to ensure the output is always syntactically and logically correct.

### ✳️ Logit Masking (The Core Mechanism) ###

When an LLM generates text, it produces a probability distribution over its entire vocabulary (logits) for the next token. 
Our algorithm intercepts these raw scores before the sampling stage:

**Identification**: 

Based on the current decoding state (e.g., extracting a number), we identify a set of "allowed" tokens (e.g., [0-9]).
    
**Masking**:

We use a constant **NEG_INF** set to $-1e11$. 
    While mathematically we aim for $-\infty$, in computing, we use a sufficiently large negative number to ensure that after the softmax operation, the probability of forbidden tokens becomes exactly $0$. We avoid using the standard "Infinity" float to prevent potential NaN (Not a Number) errors.

**Selection**:

We add this mask to the model's logits. This mathematically forces the probability of forbidden tokens to zero, ensuring the model only selects from our allowed set.
    
### ✳️ Trie-based Name Selection ###
To select a function name, we implement a Trie (Prefix Tree).All available function names are inserted into the Trie as sequences of tokens.During generation, the decoder traverses the Trie. At each step, it only allows tokens that correspond to the children of the current node.This guarantees that the model follows a path leading to a valid function name, preventing it from inventing non-existent tools.


### ✳️ Extraction: State Machine ##
The algorithm follows a structured state machine to handle different parameter types:

**Numeric State**:

Uses a regex-based token filter (^[0-9.\-eE]+$) combined with structural stop tokens (like , or }) to allow the model to finish the value.

**Boolean State**:

Restricts the vocabulary strictly to True or False.

**String State**:

Permits free generation but prevents premature closing of quotes by masking the quote token at the very first position of the value.

### ✳️ Iterative Parameter Filling ###
Rather than asking the model to generate the entire JSON at once, we use an Iterative Pipeline. We prompt the model for each parameter individually. This "Chain of Thought" style approach significantly reduces complexity for the Small Language Model (SLM).

### 🔵 Design decisions

The architecture of this project focuses on the Single Responsibility Principle.

 ### ✳️ Object-Oriented Orchestration ###

The project is organized into distinct classes, each owning a specific part of the pipeline:

* **Loader**: Manages the I/O layer. It handles functions definition and prompts files.

* **Pydantic Models**: They enforce strict schema validation and typing for all inputs, ensuring total data integrity.

* **ConstrainedDecoder**: Acting as the interface with the LLM, this class is the only one authorized to manipulate logits and handle tokenization.

* **PromptProcessor**: This is the orchestrator. It manages the sequence of calls (Function Name -> Param 1 -> Param 2).

* **StateNode**: A dedicated structure for the Trie logic. 

### ✳️ Graceful Error Handling & Recovery Layer ###

We implemented a dedicated Advanced Recovery mechanism. By isolating "repair" logic (like fixing brackets or casting types) into a separate method, we ensure the main extraction loop remains readable.

Try-except blocks prevent unexpected crashes.

## 🔵 Performance analysis

The performance of the solution was evaluated based on three main criteria:
Accuracy, Speed, and Reliability.

### ✳️ Accuracy ###

**Success rate:** 

The constrained decoding approach yields a 100% success rate for structural validity. Since forbidden tokens are mathematically excluded (masked with $-1e11$), the model cannot physically generate an invalid function name or an incorrect data type.

**Type Integrity:** 
    
By using specific masks for numbers and booleans, we eliminated the common "hallucination" where a model might return text instead of a required float.
    
**Pydantic Validation:**
    
Ensure that the final JSON is not just valid syntax-wise, but also logically consistent with the tool definitions.

### ✳️ Speed & Computational Efficiency ###

The use of constrained significantly optimizes performance:

**Logit Filtering ($V \to Vo_{allowed}$) (where $V$ is the allowed vocabulary):**

Instead of sampling from the full vocabulary ($>50,000$ tokens), the mask reduces the search space to a handful of valid tokens.
It forces the model to converge instantly on the correct value.

**Trie Efficiency ($O(L)$) (where $L$ is the length of the function name):**

Selecting function names relies on a Prefix Tree (Trie). The complexity is $O(L)$ (where $L$ is the length of the function name), making the selection time independent of how many functions are in your library.

**Inference Truncation:**
By forcing structural stop tokens (like , or }), we stop the generation when a value is complete. This prevents the model from wasting time generating useless text ("hallucinations").

**Cache Optimization ($O(1)$):** The integration of a result cache turns expensive LLM inferences into simple $O(1)$ lookups for recurring prompts.

### ✳️ Reliability: ###

Although optimized for Qwen3-0.6B, the architecture was tested with SmolLM2-360M. This ensures that the logit masking and token filtration logic are robust enough to handle different tokenizers.

## 🔵 Challenges faced

This project presented several technical and conceptual hurdles:

### ✳️ The "Starting" Challenge: ###

The most significant difficulty was starting the project. Unlike standard programming tasks, there is no "standard" way to implement constrained decoding. Finding the logical sequence of steps—from understanding how to intercept logits to deciding how to structure the iteration—required a long phase of research and several failed prototypes.

### ✳️ Prompt Engineering & Pipeline Orchestration ###
Determining the right Prompt Pipeline was a process of trial and error. I discovered that asking the LLM for a full JSON block led to frequent hallucinations.

The Challenge: Finding the instructions so the model responds optimally without being too wordy.

The Solution: I moved toward an iterative pipeline where the model is prompted for one parameter at a time. Designing this flow, ensuring the model keeps enough context to answer correctly while being restricted by our masks, was crucial for the project's success.

## 🔵 Testing strategy

To ensure the reliability of the constrained decoding pipeline, I implemented a multi-layered testing strategy combining unit tests and integration scenarios:

* **Pytest Framework**: A comprehensive suite of unit tests was developed using pytest to validate the ConstrainedDecoder in isolation. These tests verify that the logit masking correctly enforces types (numeric, boolean, string) and that the Trie-based selection never allows an invalid function name. Static analysis via mypy and flake8 further guarantees code quality and type safety.

* **Edge Case Handling (function_calling_edge_cases.json)**: I created a specific test set designed to push the model's boundaries. This includes prompts with ambiguous requests, missing information, or extreme values.

* **Extended Functionality (functions_extended.json)**: To test the flexibility of the Trie and of the decoder, I used an extended tool catalogue. This library contains functions with booleans and with complex parameter structures, confirming that the system remains performant and accurate even as the number of available tools increases.

## 🔵 Example usage

To demonstrate the power of Constrained Decoding, here is a real-world example of how the pipeline processes a request.

* **Context: Available Tools**

Imagine we have the following tool defined in functions_definition.json:

```bash
JSON
{
  "name": "set_thermostat",
  "description": "Sets the temperature in a specific room",
  "parameters": {
    "temperature": "number",
    "room": "string"
  }
}
```

* **Input Prompt**

The user provides the following unstructured request in function_calling_tests.json:

```bash
"Can you please put the heating at 22.5 degrees in the living room?"
```

* **Execution Command**
```bash
uv run python -m src --input data/input/function_calling_tests.json
```

* **Output Result**

The program generates the following structured JSON in data/output/function_calls.json. Notice how the "22.5" has been correctly extracted as a float and the room name is isolated:

```json
JSON
[
  {
    "prompt": "Can you please put the heating at 22.5 degrees in the living room?",
    "name": "set_thermostat",
    "parameters": {
      "temperature": 22.5,
      "room": "living room"
    }
  }
]
```

What happened behind the scenes?

* Function Selection: The LLM, guided by our Trie, selected set_thermostat among all available tools.

* Numeric Masking: When extracting 22.5, the decoder only allowed digits and the decimal point, preventing any textual hallucinations.

* Pydantic Validation: The result was validated against a Pydantic model before being saved to ensure the temperature is indeed a number.
