# QKV-SC: Entity-Centric Long-Term Memory for LLMs via Query-Conditioned Key-Value Semantic Caching

**QKV-SC** (Query-Key-Value Semantic Cache) is a novel long-term memory architecture for Large Language Models (LLMs) that enables entity-centric semantic storage and retrieval by directly leveraging the internal QKV (Query-Key-Value) projections of the model. This implementation is specifically optimized for **Microsoft Phi-3 Mini**, demonstrating how to extract and utilize attention mechanism weights for semantic caching without fine-tuning.

Inspired by the theoretical framework **"Categorization Is Intelligence"**, QKV-SC treats memory as the dynamic construction and retrieval of semantic categories—mirroring how intelligence itself operates through categorization.

This project is originally published at: [https://zenodo.org/records/17756627](https://zenodo.org/records/17756627)

---

## 🌟 Key Features

- **Entity-Centric Memory**: Stores facts about entities (e.g., "Zorbex") under semantic roles (`what`, `how`, `why`, `when`, `source`).
- **Direct QKV Projection Extraction**: Uses the final layer’s `qkv_proj` weight to derive `W_Q`, `W_K`, `W_V` matrices from Phi-3.
- **Semantic Role Decomposition**: Automatically parses input statements into structured semantic roles using in-context learning with Phi-3.
- **Query-Conditioned Retrieval**: Matches incoming questions to stored memories using QKV similarity in activation space.
- **Zero Fine-Tuning**: Works out-of-the-box with pretrained Phi-3 Mini (4K context).

---

## 🧠 Theoretical Foundation

This project implements a practical instantiation of the **Second Intelligence Escape** described in *"Categorization Is Intelligence"* (Wang, 2025):

> *"Intelligence reduces to the construction, maintenance, and refinement of categories. When categorization pressure exceeds substrate capacity, a phase transition occurs."*

QKV-SC externalizes transient semantic clusters (categories) from the LLM’s activation space into a persistent, queryable cache—extending the model’s effective memory beyond its context window.

The foundational theory paper is available at: [https://zenodo.org/records/17773065](https://zenodo.org/records/17773065)

---

## 🚀 Quick Start

### Prerequisites
- Python ≥ 3.9
- PyTorch
- Hugging Face `transformers`
- A local copy of **Phi-3 Mini 4K Instruct** (place in `models/phi-3-mini-4k-instruct`)

### Installation
```bash
pip install torch transformers
```

### Usage Example
```python
from qkv_sc_phi3_fixed import QKV_SC_Phi3_Fixed

# Initialize with local Phi-3 model
cache = QKV_SC_Phi3_Fixed(model_name="models/phi-3-mini-4k-instruct")

# Store a fact
cache.store_from_statement("Zorbex is good because it uses quantum chips.")

# Retrieve based on question
result = cache.retrieve("What is Zorbex?")
print(result['retrieved_text'])  # → "Zorbex is good because it uses quantum chips."
```

---

## 📂 Project Structure

```
qkv-sc/
├── qkv_sc_phi3_fixed.py       # Main implementation
└── README.md                  # This file
```

> Note: The `5.html` referenced in your request was not found. The system contains `2.html`, `3.html`, and `categorization_v1.2.html`, which provide background on the underlying theory of categorization and intelligence.

---

## 📚 Related Work & Theory

- **[Project Original Record](https://zenodo.org/records/17756627)** – Official Zenodo record for this QKV-SC implementation.
- **[Categorization Is Intelligence (v1.2)](https://zenodo.org/records/17773065)** – Foundational theory explaining the two intelligence phase transitions in Earth's history.
- **[AI’s 70-Year Evolution = 3 Categorization Paradigms](2.html)** – Historical context of AI progress through the lens of categorization.
- **[Classification Pressure and Substrate Interference](3.html)** – Application of the theory to second-language acquisition and AGI design.

---

## 📄 License

This project is open-source and available under the **MIT License**. The underlying theoretical work *"Categorization Is Intelligence"* is licensed under **CC BY 4.0**.

---

## 🙌 Acknowledgements

- Microsoft for releasing **Phi-3 Mini**
- Zhongren Wang for the foundational theory *"Categorization Is Intelligence"*
- Hugging Face for the `transformers` library

---

> **Categorization is not something intelligence does. Categorization *is* what intelligence is.**