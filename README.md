# Financial NLP : Hands-on Course
__Author__ = 'Firas A Obeid'

**Course Number/Section:** FTEC 6V96/f01  
**Course Title:** Financial NLP, Textual Analysis & Applied Research: An LLM Hands-on Course  
**Term:** Fall 2026
**Classroom:** JSOM 1.516  

**Instructor of Practice:** Firas Obeid  
**Instructor of Record:** Robert Kieschnick  

## Prerequisites

- **Admission:** MS in Financial Technology & Analytics program with completion of the machine learning sequence
- **Computer:** Laptop computer with at least 4GB of memory
- **System Requirements:** Chromium-based web browser; text editor for coding (VS Code recommended); Python 3.7+; Jupyter Notebook

## Syllabus: 

### Course Overview

This course explores the evolution of Natural Language Processing (NLP) from traditional language modeling techniques to modern Large Language Models (LLMs), using event-driven trading as an applied, hands-on laboratory theme throughout the semester. While financial news analytics and trading signal generation provide practical context for experimentation, the course is not limited to trading applications alone. Instead, the event-driven framework serves as a structured environment for students to observe how applied AI research is conducted end-to-end: from problem formulation and data collection to experimentation, model evaluation, and deployment considerations. The broader objective is to help students develop the ability to think critically about AI systems and select appropriate methodologies without becoming overwhelmed by the rapidly expanding ecosystem of tools, models, and techniques.

Throughout the course, students will work with real-world news headline data collected through Refinitiv's Reuters API to study how language models interpret, classify, and extract signals from unstructured text. Beginning with foundational NLP methods and progressing toward deep learning architectures and modern LLM systems, the course covers text classification, topic labeling, LLM design and usage patterns, fine-tuning small language models, experiment design, and advanced retrieval engine development. A central theme of the course is that classification remains one of the foundational paradigms underlying many NLP and LLM applications. Students will revisit and extend a research framework originally replicated and developed by the instructor in 2020, applying increasingly sophisticated approaches, from heuristic-based systems to fine-tuned LLMs, to classify financial news into actionable signals such as buy and sell recommendations. Beyond financial applications, the course also surveys how banks and enterprises leverage LLMs in production environments, introduces emerging agentic systems, and discusses ongoing industry research and development initiatives. Ultimately, the course emphasizes research-oriented thinking: defining problems rigorously, understanding trade-offs between techniques, and learning how to identify the right tools for a given task rather than pursuing complexity for its own sake.

### Reading Material

1. "Machine Learning for Algorithmic Trading" by Stefan Jansen
2. "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
3. "Natural Language Processing with Transformers: Building Language Applications with Hugging Face" by Lewis Tunstall, Leandro von Werra, and Thomas Wolf
4. Published research papers

---

## Course Schedule & Topics

### Lesson 1: Survey of Traditional NLP Techniques

**Date:** August 27, 2026

**Scope:** Build intuition behind rule-based NLP systems and traditional text processing pipelines.

**Technologies:**
1. Textblob
2. NLTK
3. BoW (Bag of Words)
4. TF-IDF
5. Sentiment Analysis
6. n-grams
7. Hugging Face Datasets

**Session Plan:**
- Introduce the course scope and research framework using financial news headlines from Refinitiv
- Walk through traditional NLP techniques using the financial news headlines dataset
- Build a simple lexicon-based sentiment model after defining the target variable
- Introduce Hugging Face Datasets

**Homework:** Analyze the news headlines dataset by cleaning it up and building a lexicon-based sentiment model.

---

### Lesson 2: Topic Modeling

**Date:** September 3, 2026

**Scope:** Apply unsupervised learning techniques to extract latent topics from text.

**Technologies:**
1. Probabilistic Algorithm: LDA (Latent Dirichlet Allocation)
2. Deterministic Algorithm: NMF (Non-negative Matrix Factorization)
3. FasTopic: Transformer-based Topic Modeling
4. Topic Coherence Metric

**Session Plan:**
- Extract latent topic categories from the news headlines dataset
- Explain probabilistic algorithms for latent topic modeling
- Explain deterministic algorithms for topic modeling (matrix factorization)
- Label headlines based on most important tokens for segmenting future model performance

**Homework:** Analyze the news headlines dataset to learn latent topics and label headlines accordingly.

---

### Lesson 3: Text Representation (Tokenization, Embedding & Specialized Processing)

**Date:** September 10, 2026

**Scope:** Walk through different tokenization techniques and explain the evolution of embeddings.

**Technologies:**
1. Word2Vec
2. Doc2Vec
3. n-gram tokens
4. Embeddings
5. Prominent Tokenization Methods
6. BPE (Byte Pair Encoding)
7. Byte Level Tokens

**Session Plan:**
- Demonstrate several tokenization techniques
- Walk through advanced text processing strategies with n-grams tokens
- Explain evolution of embedding vectors from Word2Vec to Doc2Vec and recent embedding models
- Explain how to use N-dimensional embeddings by averaging or normalizing them

**Formula:** L2 Normalization: emb = emb / √Σ(emb)²

**Note:** Begin searching for a research paper (not older than 3 years), read it, own it and live it!

---

### Lesson 4: Develop Your First Language Model

**Date:** September 17, 2026

**Scope:** Develop intuition behind LSTM architectures for language modeling and text representation.

**Technologies:**
1. LSTM (Long Short-Term Memory)
2. Specialized Language Models
3. Embeddings
4. Text Representation
5. Text Generation
6. Semi-supervised Learning

**Session Plan:**
- Explain differences between supervised / semi-supervised / weak / self-supervised learning
- Develop an LSTM model for training news headlines embeddings at character (byte) level to token level
- Develop an LSTM model for text generation
- Explain the notion of LSTMs for representation vs generation

---

### Lesson 5: Project 1 (Warm-Up Project)

**Date:** September 24, 2026

**Description:** Build a story from homework assignments 1 & 2 by analyzing news headlines tokens, topics, token statistics, news headline statistics, etc. This serves as a warm-up for your final project.

---

### Lesson 6: Fine Tune the LSTM for News Headline Classification

**Date:** October 1, 2026

**Scope:** Fine-tune the LSTM language model from Lesson 4 on a downstream classification task for news headline buy/sell predictions.

**Technologies:**
1. Transfer Learning
2. LSTM
3. Fine-Tuning
4. Language Modeling
5. Text Decoding Strategies
6. Hugging Face (Spaces)
7. Gradio Deployments

**Session Plan:**
- Fine-tune the LSTM language model on a classification task
- Capture model performance using targeted business metrics
- Explain random vs. fixed sampling from the original language model
- Demo Hugging Face deployment of the event-driven LSTM for trading

---

### Lessons 7-8: Alternative & Exhaustive Text Classification Techniques

**Expected Date/s:** October 8-15, 2026

**Scope:** Survey modern classification techniques to create competing models with the previously developed LSTM classifier.

**Technologies:**
1. Experiment Design
2. CBOW (Continuous Bag of Words)
3. Skip-gram
4. FastText
5. In-context Learning
6. Skrub
7. Natural Language Inference (NLI)
8. Zero-shot Classifiers
9. Embedding Models (Encoders)
10. Cosine Similarity with SVC
11. Advanced Retrieval Systems
12. Text-based Similarity Metrics (non-semantic)
13. LLM Prompting for Experiment-based Scientific Discoveries

**Session Plan:**
- Develop intuition behind all techniques
- Develop models and compare with the LSTM language model trained on news headlines classification
- Develop intuition behind similarity measures for classification
- Discuss how natural language inference works
- Demonstrate advanced retrieval systems

---

### Lessons 9: Survey Encoders / Decoders Model Design & Usage

**Expected Date/s:** October 22-29, 2026

**Scope:** Explain various encoder/decoder architectures and how each LLM paradigm works from input/output perspective.

**Technologies:**
1. Self-Supervised Learning
2. Encoders
3. Decoders (Catalog)
4. MLM / NSP / SOP / NLI
5. BERT / E5
6. Cross-Encoders
7. Hugging Face Model Loaders
8. Attention Mechanisms
9. Contrastive Learning
10. Decoder Architecture Manipulation Survey
11. Miscellaneous LLM Topics
12. Financial LLM Topics

**Session Plan:**
- Explain how encoders, decoders, and encoder-decoders work (input/output and high-level inner components)
- Show how self-supervision manifests in encoder training
- Explain different encoder training paradigms
- Demo how to use several model instances from Hugging Face
- Discuss encoder vs. decoder architectures for classification

---

### Lessons 10: LLM Fine-Tuning for News Headline Classification

**Expected Date/s:** November 5-12, 2026

**Scope:** Further expand the classification leaderboard by fine-tuning an open-source LLM and compare results with previous classifiers.

**Technologies:**
1. Fine-tuning
2. PEFT (Parameter-Efficient Fine-Tuning)
3. Adapters / LoRA / QLoRA
4. Quantization
5. Fine-tune Encoder / Decoder
6. SLMs (Small Language Models) for Edge Deployments
7. Emerging Fine-Tuning Methods
8. Fine-Tuning as a Service
9. Hugging Face Trainer

**Session Plan:**
- Build intuition around using LLMs to fine-tune
- Discuss the paradigm from in-context learning to fine-tuning on the other end of the scale
- Discuss QLoRA and attention mechanism variants
- Demo fine-tuning Google's Gemma LLM for classifying the news headlines dataset

---

### Lesson 11: Special Topics

**Expected Date/s:** November 19 - December 3, 2026

**Scope:** Discuss special selected topics from recent research papers and agentic approaches.

**Topics:**
1. **Technical Special Topics:** PyTorch vs TensorFlow, Uncertainty Quantification
2. **Financial Special Topics & Institutional Use Cases:** Credit, Fraud, and Market Risk (Application-Focused)
3. **Agentic Systems:** Levels of LLM Usage Agency (Tools, MCPs, Coding Agents, Autoresearch)

**Session Plan:**
- Explain a strategy for training data selection for LLMs
- Discuss Google's white paper on LLMs
- **LLM Function Calling:** Demonstrate how to use Google's Gemini/Gemma LLM to retrieve news headlines for stocks with the highest VWAP, using external APIs, on the same day predictions are made
- Survey emerging agentic systems and their applications in finance

---

### Lessons 12-13: Project 2 (Final Project)

**Expected Date/s:** December 3-17, 2026

**Description:** Applied research project with hands-on mentorship in the final 2 Lessons.

**Project 2 Grading Breakdown:**
- **Presentations:** 15 pts
- **Deployment:** 15 pts
- **GitHub/Hugging Face Documentation:** 10 pts
- **Research Paper Replication:** 10 pts (waived if project is based on published paper)
- **Execution (Idea/Code):** 50 pts

**Note:** If Project 2 is based on a published research paper, you will automatically receive +10 bonus points and waive the separate research paper assignment.

---

## Grading Breakdown

| Component | Percentage |
|-----------|-----------|
| Homework | 20% |
| Project 1 | 30% |
| Project 2 | 40% |
| Research Paper Results Replication | 10% |

**Note:** Project 2 and the Research Paper Replication can be combined.

---

## Course & Instructor Policies

### Communication
- **Platform:** eLearning will be used for class content (slides, assignments) and grade recording
- **Slides:** Posted before each class
- **Announcements:** Posted for changes in assignment Expected Date/s
- **Instructor Response Policy:** All student inquiries (emails, voice messages) will be answered within 48 hours (excluding holidays and Lessonends)
- **Instructor NOTE** All material will be shared via github commits,  where there will be a google drive link for the Lesson plans folder that is restricted too student access only

### Attendance Policy
Attendance is extremely important. Students are expected to attend all classes to achieve maximum success. Attendance will be taken and considered for the participation grade. The instructor's judgment of the value of contributions to class discussion will also be reflected in this grade.

**Note:** There is no makeup for missed in-class assignments.

### International Students
The University's attendance policy requires international students to attend class in person. However, under certain conditions, online participation may be acceptable if situations are limited.

### Class Recordings
- The instructor may record course meetings
- Recordings are available to all registered students to supplement classroom experience
- Students must follow University policies and maintain security of access passwords
- **Students are prohibited from recording any part of this course** unless approved by the Office of Student Accessibility
- Recordings cannot be published, shared with those outside the class, or uploaded elsewhere without approval
- Violations constitute a breach of the Student Code of Conduct

### Late Work Policy
All assignments are due at 11:00 PM on the specified date. Late assignments are not accepted unless prior arrangements have been made with the instructor.

### Academic Integrity
The University is committed to academic excellence and expects academic honesty from all members of the community. Academic honesty includes:
- Adherence to instructor-established guidelines for individual and group work
- **Prohibition of plagiarism:** Representing others' work as your own
- **Prohibition of cheating:** Receiving unauthorized aid on assignments
- **Prohibition of reuse:** Using similar papers or work products for different classes without instructor permission

**Penalties:** Grade of "F" on the work or course, plus disciplinary action per University policy.

### Working Together on Individual Assignments
Each student is expected to do their own work on individual assignments. Copying another student's work (computer files) or having another person do your work is scholastic dishonesty and will be dealt with accordingly.

---

## Additional Resources & Code Scope

### Learning Outcomes
By the end of this course, students will be able to:
- Build hands-on intuition behind text processing and fundamental NLP libraries (NLTK, gensim, spacy)
- Build several tokenizers and encoding schemes
- Host training and validation sets on Hugging Face
- Introduce several Hugging Face LLM pipelines (classification and embedding)
- Develop a language model locally and push it to Hugging Face for usage through the transformers SDK
- Deploy locally developed language models on Hugging Face Spaces as Gradio apps
- Demonstrate the same language model in both TensorFlow and PyTorch versions
- Demo LLM fine-tuning through the Trainer API and push results to Hugging Face
- Develop a framework using function calling for event-driven trading based on price movement heuristics and filtering rules

### External Resources

* **[AI Hedge Fund Repository](https://github.com/virattt/ai-hedge-fund)**
    - Market Data Agent: Gathers market data (stock prices, fundamentals)
    - Quant Agent: Calculates technical signals (MACD, RSI, Bollinger Bands)
    - Fundamentals Agent: Analyzes profitability, growth, financial health, valuation
    - Sentiment Agent: Examines insider trades for sentiment
    - Risk Manager: Determines risk metrics (volatility, drawdown)
    - Portfolio Manager: Makes final trading decisions and generates orders

* **[Gemini API Documentation](https://ai.google.dev/gemini-api/docs/models/gemini#text-embedding)**

* **[Fine-Tune Gemma using Hugging Face Transformers and QLoRA](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)**

* **[Data Science Agent](https://developers.googleblog.com/en/data-science-agent-in-colab-with-gemini/)**

* **[Hugging Face Papers](https://huggingface.co/papers)** - For finding recent research papers

---

## Course Citation

```bibtex
@inproceedings{dos-santos-pinheiro-dras-2017-stock,
    title = "Stock Market Prediction with Deep Learning: A Character-based Neural Language Model for Event-based Trading",
    author = "dos Santos Pinheiro, Leonardo and Dras, Mark",
    editor = "Wong, Jojo Sze-Meng and Haffari, Gholamreza",
    booktitle = "Proceedings of the Australasian Language Technology Association Workshop 2017",
    month = dec,
    year = "2017",
    address = "Brisbane, Australia",
    url = "https://aclanthology.org/U17-1001/",
    pages = "6--15"
}
```

---

## General Policies & Procedures

For information regarding general University policies and procedures, please visit [go.utdallas.edu/syllabus-policies](https://go.utdallas.edu/syllabus-policies). These policies include:
- Technical Support
- Field Trip Policies, Off-Campus Instruction and Course Activities
- Student Conduct and Discipline
- Academic Integrity
- Copyright Notice
- Email Use
- Withdrawal from Class
- Student Grievance Procedures
- Incomplete Grade Policy
- Disability Services
- Religious Holy Days
- Avoiding Plagiarism


## ⚠️ Commercial Usage & Alpha Protection Notice
Any use, adaptation, deployment, or derivative implementation of these ideas, 
frameworks, or code architectures for commercial gain is STRICTLY PROHIBITED 
under the default terms of this repository. 

Prohibited commercial activities include, but are not limited to:
- Designing, teaching, or selling commercial courses, bootcamps, workshops, 
  seminars, or professional certificates.
- Integrating these code architectures, research models, or pedagogical 
  frameworks into commercial software or commercial training products.
- Deploying or adapting these quantitative models, strategies, mathematical 
  structures, or logic to manage capital, execute trades, or generate trading 
  alpha at any financial institution, hedge fund, proprietary trading firm, 
  asset management company, or algorithmic trading desk.


## Professor/Author Citation
```yaml
cff-version: 1.2.0
message: "If you use, adapt, or build upon the ideas in this syllabus, please cite it as follows."
authors:
  - family-names: Obeid
    given-names: Firas A
    affiliation: University of Texas at Dallas
title: "UT Dallas Course Syllabus: Financial NLP, Textual Analysis & Applied Research: An LLM Hands-on Course "
date-released: 2026
url: "https://github.com/firobeid/UTD_FINTECH_NLP_COURSE"
```