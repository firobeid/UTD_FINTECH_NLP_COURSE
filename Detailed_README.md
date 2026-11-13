# Financial NLP for Trading: Hands-on Course Outline

## Course Overview:
The course focuses on language models and NLP techniques specifically for event-driven trading. While many techniques are generalizable, the core application remains financial trading. It revisits traditional NLP methods and progresses to the latest advancements in language modeling. A key element is the consistent use of a news headlines dataset from Refinitiv's (Reuters) API throughout the course to apply various language modeling approaches. The course emphasizes classification as the primary output mechanism for NLP models and is structured around replicating and developing a language model from a 2020 research paper. This model is then challenged to classify news headlines (Buys/Sells) using various deep learning models, from simple heuristics to fine-tuning off-the-shelf Large Language Models (LLMs), and finally introducing new techniques with classification as the core paradigm.

## Weekly Topics and Technologies Covered:

### Week 1: Survey of Traditional NLP Techniques
- **Scope:** Establish the course workflow using the Refinitiv/Reuters headlines dataset, then walk through NLP 101 from raw text cleanup to baseline sentiment signals. Students start with tokens, n-grams, and dictionary-based methods to understand how unstructured text is transformed into model-ready features.
- **Technologies:** TextBlob, NLTK, Bag-of-Words (BoW), TF-IDF, lexicon-based sentiment scoring (Loughran-McDonald style), n-grams, Hugging Face Datasets.
- **Key Learnings:** Build a strong baseline before deep learning by understanding tokenization, feature extraction, and sentiment scoring trade-offs. Learn to frame the Buy/Sell label problem clearly, validate data assumptions, and create reproducible first-pass NLP experiments.

### Week 2: Topic Modeling: Unsupervised Techniques for Retrieving Latent Topics
- **Scope:** Move from labeled prediction to unsupervised discovery by extracting latent news themes and segmenting the headline universe into interpretable groups. The session compares probabilistic, matrix-factorization, and transformer-assisted topic modeling approaches.
- **Technologies:** LDA (Latent Dirichlet Allocation), NMF (Non-negative Matrix Factorization), FASTopic, topic coherence workflows (segmentation + confirmation + aggregation), CountVectorizer/BoW representations.
- **Key Learnings:** Learn how different topic models represent signal structure, how to evaluate topic quality with coherence, and how to assign practical topic labels that later support per-topic classifier evaluation.

### Week 3: Text Representation (Advanced Processing, Tokenization & Embeddings)
- **Scope:** Deepen representation learning by moving from sparse text vectors to dense embeddings and by comparing tokenization strategies (word, subword, and byte-level). Students train custom Word2Vec/Doc2Vec-style representations and inspect how vector spaces encode semantic proximity.
- **Technologies:** Word2Vec (CBOW/Skip-gram), Doc2Vec, n-gram tokenization, BPE and byte-level tokenization, embedding matrices, similarity metrics.
- **Key Learnings:** Understand when non-contextual, limited-context, and contextual representations are appropriate. Learn how tokenizer design impacts downstream model quality, and how to convert embeddings into robust features for later classifiers.

### Week 4: Language Modelling (Develop our First Language Model)
- **Scope:** Build the first end-to-end language model and connect representation learning to generation. The week introduces sequence modeling paradigms (stateful vs stateless) and demonstrates character-level and word-level modeling for completion and generation tasks.
- **Technologies:** LSTM language models, embeddings, sequence modeling, text generation/decoding strategies, supervised vs semi/self-supervised training perspectives.
- **Key Learnings:** Learn why language modeling is foundational for transfer learning, how hidden-state memory shapes context handling, and how generation behavior changes with different sampling/decoding decisions.

### Week 5: Project 1
- **Scope:** Consolidate Weeks 1-4 into a mini applied-research narrative that connects data quality, topic structure, and baseline model behavior.
- **Technologies:** Data profiling, token/topic statistics, exploratory visualization, baseline sentiment and representation outputs from prior lessons.
- **Key Learnings:** Practice translating technical outputs into an evidence-based story, identify error patterns early, and define a sharper direction for the final project pipeline.

### Week 6: Language Models For Classification (Transfer Learning, Fine Tuning & Deployment)
- **Scope:** Convert the language model from Week 4 into a downstream Buy/Sell classifier, then operationalize it for live inference. Emphasis is placed on transfer learning, fine-tuning protocols, evaluation discipline, and deployment readiness.
- **Technologies:** Transfer Learning, fine-tuning workflows, ULMFiT-style adaptation, decoding strategy comparisons, Hugging Face Spaces, Gradio deployment.
- **Key Learnings:** Learn how pretrained language dynamics transfer into classification, how to measure model utility with business-aware metrics, and how to package an experiment into a usable demo endpoint.

### Week 7/8: A Survey of Alternative Text Classification Techniques (Develop Contender Financial News Headlines Classifiers)
- **Scope:** Build a rigorous model leaderboard by testing classical, embedding-based, transformer-based, and foundation-model-style alternatives against the LSTM benchmark. In parallel, develop an advanced retrieval system through an ablation framework that isolates the contribution of each retrieval component (candidate generation, decision function, pooling strategy, and reranking) on downstream Buy/Sell classification quality.
- **Technologies:** Dummy baselines, FastText, lexicon-based classifiers, string kernels, similarity measures, headline-embedding extraction pipelines, SIF (Smooth Inverse Frequency) embedding pooling, similarity-based classifiers (and enhanced variants), SVC as the decision function for retrieval-driven classification, reranking/re-ranking pipelines, embedding-as-features workflows, FinBERT, NLI/zero-shot classification, Skrub encoders, TabPFN, and TabICL.
- **Key Learnings:** Learn how to design and interpret ablation studies for advanced retrieval systems, quantify what each module adds to precision/recall and ranking quality, and understand when string-kernel signals, SIF pooled embeddings, SVC decision boundaries, and reranking logic produce robust gains for financial headline intelligence.

### Week 9: Survey Encoder, Decoder & Encoder-Decoder Language models (Self-supervised learning, MLM, NSP & more)
- **Scope:** Highly technical deep-dive into the anatomy of transformer systems and the core design innovations behind modern LLMs. The session dissects each major component (token/position embeddings, attention blocks, residual pathways, normalization, MLP/FFN modules, output heads), explains how these pieces interact through the forward pass, and compares how encoder-only, decoder-only, and encoder-decoder families structure inputs/outputs for different objectives.
- **Technologies:** Self-supervised pretraining objectives (MLM, NSP/SOP, causal next-token prediction), RNN/GRU encoder-decoder foundations, Bahdanau/Luong attention, transformer QKV attention mechanics, attention variants and modern updates in current LLMs (efficiency/architecture adaptations), residual and block design patterns, contrastive learning setups, NLI-style supervision framing, loss-function design choices across generation/retrieval/classification, and Hugging Face model loaders (PyTorch/TensorFlow) for architecture inspection.
- **Key Learnings:** Students develop component-level intuition for why transformer performance depends on architecture + objective + I/O formulation, not just model size. They learn how changes in tokenization, sequence formatting, label construction, and loss design become primary levers for LLM capability, and how modern LLM updates often come from restructuring attention/blocks/objectives to improve reasoning quality, context handling, efficiency, and downstream task transfer.

### Week 10: Fine-Tune on LLM for News Headline Classification
- **Scope:** Highly technical deep-dive on post-pretraining adaptation mechanics for financial headline classification. The session formalizes the full usage continuum (in-context learning, prompt engineering, RAG, PEFT, full fine-tuning), then analyzes what is actually being updated in each paradigm and how those update choices affect optimization stability, generalization, and inference efficiency.
- **Technologies:** Fine-tuning objective design (classification vs generative label emission), SFT data structuring and label-format design, parameter-update strategies (feature extraction, partial unfreezing, full-model updates), PEFT methods (LoRA/QLoRA/prefix tuning/adapters), layer-targeting decisions (attention/MLP/embeddings), post-training modularity and MoE-style branching ideas, quantization and QAT trade-offs, pruning, distillation/self-distillation pipelines, hallucination-aware SFT framing, encoder fine-tuning tracks (ModernBERT/UltraBERT/RoBERTa-style), decoder fine-tuning tracks (Gemma-family), and Hugging Face training stack orchestration.
- **Key Learnings:** Students learn to treat fine-tuning as an optimization-and-representation engineering problem: align task I/O format with objective/loss, choose update scope based on compute and risk constraints, and evaluate adaptation quality beyond accuracy using latency, memory, robustness, and calibration signals. They leave with a framework for selecting the right fine-tuning strategy for institution-grade financial NLP deployment.

### Week 11: Special Topics
- **Scope:** Financial institutional LLM use-cases are the center of this lesson. The session surveys how banks, payment firms, asset managers, and market institutions apply LLMs/foundation models in production workflows across research, risk, compliance, forecasting, and transaction intelligence.
- **Technologies:** Institutional LLM application patterns and case studies across macro/central-bank communication analysis, expert-judgment replication, disclosure and sentiment tracking, fraud and payments intelligence, credit/transaction foundation models (e.g., Stripe, Revolut/PRAGMA, Visa/TREASURE, Plaid, Affirm), plus governance and deployment controls used in regulated environments (SEC/FINRA/Fed/UK context).
- **Key Learnings:** Students understand how financial firms operationalize LLMs end-to-end: selecting use-cases, choosing model/data strategies, balancing accuracy vs latency/cost, and implementing governance, auditability, and risk controls for institution-grade deployment.

### Week 12/13: Project 2 (Final Project)
- **Scope:** Deliver an end-to-end applied research project that integrates data, modeling, evaluation, deployment, and communication. The best-performing classification strategy is wrapped in an MCP/function-calling style workflow for paper-trading simulation.
- **Technologies:** Hugging Face datasets/models/spaces, model serving patterns, API/function-calling integration, experiment tracking, project documentation standards.
- **Key Learnings:** Demonstrate independent research execution, justify architectural choices with evidence, and present a production-minded prototype that is reproducible, testable, and clearly documented.
- **Grading Breakdown:** Presentations (15pts), Deployment (15pts), GitHub/Hugging Face Documentation (10pts), Research Paper Replication (10pts), Execution (Idea/Code) (50pts).
- **Note:** Project 2 and the research paper can be combined. If Project 2 is based on a published research paper, you automatically get +10pts and waive the separate research paper assignment.

## Grading Breakdown:
- Homework: 20%
- Project 1: 30%
- Project 2: 40%
- Research Paper Results Replication: 10%

## Code Scope:
- Hands-on intuition for text processing and fundamental NLP libraries (NLTK, gensim, spacy).
- Building various tokenizers and encoding schemes.
- Hosting training and validation sets on Hugging Face.
- Introducing several Hugging Face LLM pipelines (classification & embedding).
- Developing LMs locally and pushing to Hugging Face models for use via the "transformers" SDK.
- Deploying locally developed LMs on Hugging Face Spaces as a Gradio app.
- Showing TensorFlow version of the same LM in PyTorch.
- Demonstrating LLM fine-tuning through the "Trainer" API and pushing to Hugging Face transformers.
- Revealing a framework using function calling for event-driven trading based on price movement heuristics filtering rules.

## Uniqueness from other AI in Finance courses:

This course stands out due to its:

- **Hands-on, Practical Focus:** The emphasis on "hands-on" learning, including replicating a research paper, developing and fine-tuning language models, and deploying them as Gradio apps, provides a practical skill set directly applicable to quantitative finance.
- **Event-Driven Trading Specialization:** Unlike general AI in finance courses, this curriculum is specifically tailored to event-driven trading, using real-world news headlines data for classification tasks (Buys/Sells).
- **Comprehensive NLP Coverage:** It covers a broad spectrum of NLP techniques, from traditional methods (Textblob, NLTK, BoW, TF-IDF) to advanced language modeling (LSTMs, Transformers, BERT, LLMs like Gemini/Gemma), ensuring a deep understanding of the field.
- **Integration of Cutting-Edge Technologies:** The course incorporates modern tools and platforms such as Hugging Face Datasets, Spaces, and Trainer, along with concepts like Transfer Learning, PEFT, Adapters, and QLoRA, keeping the content highly relevant to current industry practices.
- **Focus on Classification as a Core Paradigm:** The consistent theme of classification as the output mechanism for NLP models in a trading context provides a clear and actionable framework for applying these techniques.
- **Real-world Data Application:** The use of a news headlines dataset from Refinitiv's (Reuters) API throughout the course grounds the theoretical concepts in practical, financial data.
- **Emphasis on Model Deployment:** The inclusion of deploying models on Hugging Face Spaces via Gradio highlights the importance of operationalizing machine learning models, a crucial aspect often overlooked in academic settings.
- **Exploration of Special Topics:** The inclusion of topics like PRESELECT for LLM data selection and Function Calling for a trading framework demonstrates an awareness of advanced, specialized areas within financial NLP.
- **Backend Project Development:** The course includes the development of a backend project, culminating in wrapping the best classification technique in an MCP or function calling framework for paper trading, providing a complete end-to-end project experience.