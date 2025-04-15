# [Project Title]

[Briefly describe the goal and scope of your project.]

## Table of Contents

1. [Project Description](#project-description)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [Running the Project](#running-the-project)
5. [Reproducibility](#reproducibility)
6. [Team Contributions](#team-contributions)
7. [Results & Evaluation](#results--evaluation)
8. [References](#references)

---

## Project Description: Document-level Information Extraction using NLP and LLMs

[Provide a clear and concise summary of your project, outlining the problem statement, objectives, and your chosen approach using NLP and LLM techniques.]

### Problem Statement

In real-world documents, critical information is often scattered across multiple sentences and paragraphs, making it challenging to extract structured knowledge at scale. The **Document-level Information Extraction (DocIE)** challenge addresses this problem by requiring models to identify **entities**, their **mentions**, and the **relationships** between them across an entire paragraph or document, rather than at the sentence level.

The challenge is divided into two key tasks:

- **Task 1: Named Entity Recognition (NER)** – Identify all entity mentions and classify them into predefined entity types across the paragraph.
- **Task 2: Relation Extraction (RE)** – Identify and classify all relationships between relevant pairs of entities, considering paragraph-level context.

### Objectives

Train a model which is able to:
- Accurately identify all mentions of entities within a paragraph.
- Classify these entities into appropriate categories (NER).
- Detect semantic relationships between entity pairs (RE).
- Perform well in a document-level setting, where entities and relations span multiple sentences.

### Approach

#### 1. **Data Preprocessing**
- Load the training data into memory
- Combine all files to one unified dataset
- Clean & standardize
    - Normalize text: special characters, lower-case, remove formatting
    - Standardize entities (better for grouping)

#### 2. **Baseline Evaluation**
- Baseline provided by challenge
- Done with GPT-4o and llama3-ob-all
- Relevant: F1, Precision, Recall

#### 3. **Entity and Mention Extraction (Task 1 - NER)**
- Model Input Preparation:
    - Find tokenizer for long contexts
    - Span-based Annotation: Prepare training examples that mark every possible span and associate a label using a start–end classifier that directly predicts whether a span is a valid entity and its type
    - Entity Grouping / Coreference Resolution: Use clustering based on contextual embeddings (from the transformer). Group spans that refer to the same entity so that each output entity consists of its set of mentions along with an overall entity type.
    - Model Training: Train the span-based NER model end-to-end on the training data. Use standard loss functions (e.g., cross-entropy) over candidate spans and add additional losses if you choose to jointly train entity grouping via multi-task learning.
    - Output for NER: For each document, output:
        - A set of mention clusters, where each cluster has one or more spans extracted from the text.
        - The predicted entity types for those clusters.

#### 4. **Relation Extraction (Task 2 - RE)**
- Get entity span representations from task 1
- Form span pairs (E1, E2)
    - For each pair of entity spans (h_i, h_j), where i ≠ j:
    - Skip invalid or redundant pairs (e.g., same span, wrong types, overlapping spans if needed).
- Contextualize Pairs via Attention
    - Cross-Attention + Pairwise Aggregation (computationally less expensive)
        <ol>
            <li>Compute attention weights between each span and all document tokens.</li>
            <li>Aggregate attended context for each entity span</li>
            <li>Concatenate</li>
        </ol>
- Training the RE Model:
    - Utilize the labeled relation triples (extracted from the triples field) to supervise the classification.
    - Include a “no_relation” class in cases where an entity pair does not have a true relation.
    - Ensure the model learns to use both the local information (the entities’ embeddings) and the global document context.
- Output for RE: The predicted relation triples, each linking a pair of entities (typically represented by their grouped mentions) to a relation label.

#### 5. **Post-Processing and Final Output Structure**
- Entity and Relation Linking:
    - Integrate the outputs of NER and RE to produce a final output structure per document:
        - The complete set of entities (with groups of mentions and types).
        - A set of relation triples linking these entities.
- Error Analysis & Refinement:
    - Use standard metrics (e.g., Precision, Recall, F1) for both NER and RE.
    - Analyze common errors (e.g., false positives in relation extraction, misaligned entity boundaries) to adjust the grouping algorithm or the attention mechanism.
    - Optionally, apply domain adaptation techniques if performance varies significantly across domains.

## Project Structure

```
.
├── data/
│   ├── raw/                  # Raw dataset
│   └── processed/            # Processed dataset
├── logs/                     # Logs of training and evaluation runs
├── metrics/                  # Evaluation metrics and results
├── models/                   # Model checkpoints and exports
│   ├── checkpoints/
│   └── MyFirstModel.onnx
├── utils/
│   └── trainingMyCrazyModel.py
├── .gitignore
├── 1_Preprocessing.ipynb
├── 2_Baseline.ipynb
├── 3_Training.ipynb
├── 4_Evaluation.ipynb
├── 5_Demo.ipynb
├── CLEANCODE.MD
├── HELP.MD
├── README.MD
└── requirements.txt
```

---

## Setup Instructions

> [!NOTE]  
> This is only a Template. And you can add notes, Warnings and stuff with this format style. ([!NOTE], [!WARNING], [!IMPORTANT] )

### Clone Repository
```bash
git clone [repository-url]
cd [repository-folder]
```

### Create Environment
```bash
python -m venv venv
source venv/bin/activate  # Unix or MacOS
venv\Scripts\activate     # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Project

Follow these notebooks in order:
1. `1_Preprocessing.ipynb` - Data preprocessing
2. `2_Baseline.ipynb` - Establishing a baseline model
3. `3_Training.ipynb` - Model training
4. `4_Evaluation.ipynb` - Evaluating model performance
5. `5_Demo.ipynb` - Demonstration of the final model

You can also run custom scripts located in the `utils/` directory.

---

## Reproducibility

- **Random seeds:** Make sure random seeds are set and noted in your notebooks.
- **Environment:** Include the exact versions of libraries used (already covered by `requirements.txt`).
- **Data:** Clearly state sources of your data and any preprocessing steps.
- **Model Checkpoints:** Provide checkpoints clearly named and explained.

---

## Team Contributions

| Name              | Contributions                                  |
|-------------------|------------------------------------------------|
| Daniel Locher     | Data preprocessing, baseline model evaluation. |
| Nina Krebs     | Model training, hyperparameter tuning.         |

*[Each team member should describe their contributions clearly here.]*

---

## Results & Evaluation

- [Briefly summarize your evaluation metrics, improvements from baseline, and insights drawn from experiments.]
- All detailed results are documented in `metrics/firstResults.json`.

---

## References

[List here any relevant papers, sources, libraries, or resources used in your project.]

- Doe, J. (2024). *Great NLP Paper*. Conference Name.
- [Library Used](https://example-library.com)

---
