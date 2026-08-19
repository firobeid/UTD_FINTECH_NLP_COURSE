# Financial NLP for Trading: Hands-on Course Outline

## Course Overview:
The course focuses on language models and NLP techniques specifically for event-driven trading. While many techniques are generalizable, the core application remains financial trading. It revisits traditional NLP methods and progresses to the latest advancements in language modeling. A key element is the consistent use of a news headlines dataset from Refinitiv's (Reuters) API throughout the course to apply various language modeling approaches. The course emphasizes classification as the primary output mechanism for NLP models and is structured around replicating and developing a language model from a 2020 research paper. This model is then challenged to classify news headlines (Buys/Sells) using various deep learning models, from simple heuristics to fine-tuning off-the-shelf Large Language Models (LLMs), and finally introducing new techniques with classification as the core paradigm.

## Weekly Topics and Technologies Covered:

### Week 1: Survey of Traditional NLP Techniques
- **Scope:** Introduce the scope of the course, the news headlines dataset, Natural Language Processing 101, tokens & n-grams, traditional NLP techniques, sentiment analysis using a lexicon-based approach, and Hugging Face Datasets.
- **Technologies:** Textblob, NLTK, Bag-of-Words (BoW), TF-IDF, Sentiment Analysis, n-grams, Hugging Face Datasets.
- **Key Learnings:** Revisit traditional NLP techniques from early stages to most recent advancements. The course is centered around a news headlines dataset. Build intuition of language treatment and evolve this understanding as the sessions progress. Start with basic text breakdown and create simple lexicon-based models.

### Week 2: Topic Modeling: Unsupervised Techniques for Retrieving Latent Topics
- **Scope:** Familiarization with unsupervised learning techniques for learning latent topics.
- **Technologies:** Probabilistic Algorithm: LDA, Deterministic Algorithm: NME, FastTopic, Segmentation.
- **Key Learnings:** Extract latent topic categories from the news headlines dataset, explain probabilistic and matrix factorization algorithms for topic modeling, and label news headline categories based on important tokens.

### Week 3: Text Representation (Advanced Processing, Tokenization & Embeddings)
- **Scope:** Introduce dense vector representations that incorporate more semantic on limited contextual meaning behind tokens in their sentence. Learn about one text processing setup to prepare a vocab dictionary. Train Word2Vec and Doc2Vec models to extract embeddings that can be used as features in classification models. Survey various tokenization techniques other than word-level tokens.
- **Technologies:** Word2Vec, Doc2Vec, N-grams, Embeddings, Tokenization.
- **Text Representation Paradigms:** Non-Context Dependent (BoW, TF-IDF, One-Hot Encoding), Limited Context Dependent (Word2Vec, CBOW, Skip-gram, GLoVe, FastText), Full Contextual Representation (Bidirectional LSTMs (ELMo), Encoder-Based (BERT), Decoder-Based (GPT), Transformers).

### Week 4: Language Modelling (Develop our First Language Model)
- **Scope:** Learn about the premise behind language modeling. Introduce various learning disciplines such as supervised, semi-supervised learning, and more. Revisit embeddings by training your own through a semi-supervised language modeling scope to capture more contextual relationships. Use LSTMs to build language models for text representations, text generation, and text completion.
- **Technologies:** LSTMs, Language Models, Embeddings, Text Generation, Semi-supervised Learning.
- **Historical Context:** Review the evolution of NLP systems from rule-based (19xx's) to traditional tabular-based ML and shallow NN classifiers (20xx's), to Seq2Seq LMs and simple attention mechanisms (2014), to Attention(Self) Is All You Need & Transformers (2017), and modern Language Models (Transformers and State Space Models) (2018+).

### Week 5: Project 1
- **Description:** Build a story from Homework 1 and 2 by analyzing news headlines tokens, topics, token statistics, and news headline statistics. This serves as a warm-up for the final project.

### Week 6: Language Models For Classification (Transfer Learning, Fine Tuning & Deployment)
- **Scope:** Develop your first language model based classifier. Introduce various learning disciplines such as transfer learning and fine-tuning. Revisit language modeling and reflect on how LMs model an input news headline probability. Language models for text generation can have various decoding strategies. Deploy your fine-tuned event-driven language model as a Gradio web application for live inference.
- **Technologies:** Transfer Learning, Fine Tuning, ULMFiT, Text Generation Decoding Strategies, Hugging Face Spaces, Gradio.
- **Key Learnings:** Understand the concept of Transfer Learning and its application. Explore different text generation decoding strategies.

### Week 7/8: A Survey of Alternative Text Classification Techniques (Develop Contender Financial News Headlines Classifiers)
- **Scope:** Visit various text classification techniques. Recall a survey of text classifiers from early days to most recent advancements. The presented classification techniques will be contenders to the existing LSTM-based language model. Develop a leaderboard and compare several approaches' performance on the overall dataset and within the topics identified in Lesson 2. This session is hands-on motivated.
- **Technologies:** Dummy Classifier, Bag of Tricks for Efficient Text Classification, Lexicon-based sentiment classifier, Similarity Measures, Non-context dependent Classifiers, Text Classification using String Kernels, Word2Vec/Doc2Vec Classifier, FastText, LM-LSTM Classifier, Skrub (Pretrained Encoders Sentence Embedding Extraction), FinBERT (Financial BERTransformer), ZeroShot Classifiers (Natural Language Inference), TabPFN (Foundation model for tabular data), TabICL (Foundation model for in-context learning).
- **Key Learnings:** Explore a wide array of classification techniques, from simple to advanced, and understand their application in financial news analysis.

### Week 9: Survey Encoder, Decoder & Encoder-Decoder Language models (Self-supervised learning, MLM, NSP & more)
- **Scope:** Develop your first language model based classifier. Introduce various learning disciplines such as transfer learning and fine-tuning. Revisit language modeling and reflect on how LMs model an input news headline probability. Language models for text generation can have various decoding strategies. Deploy your fine-tuned event-driven language model as a Gradio web application for live inference.
- **Technologies:** Self-supervised learning, RNNs (Encoder-Decoder Era), Attention Mechanism, Encoders, Masked Language Modelling (MLM), Next Sentence Prediction (NSP), Natural Language Inference, Decoders, Prominent Model Examples, Hugging Face Model loaders.
- **Key Learnings:** Understand the different architectures of modern language models and their pretraining tasks.

### Week 10: Fine-Tune on LLM for News Headline Classification
- **Scope:** Further expanding the classification leaderboard by fine-tuning an open-source LLM and comparing results with previous classifiers/language models.
- **Technologies:** Fine-tuning, PEFT, Adapters, QLoRA, Hugging Face Trainer.
- **Key Learnings:** Build intuition for using LLMs to fine-tune, discuss in-context learning vs. fine-tuning, discuss QLoRA and attention mechanism variants, and demonstrate fine-tuning Google's Gemma 3 LLM for news headline classification.

### Week 11: Special Topics
- **Scope:** Discussing selected topics from recent research papers and an agentic approach to trading from the best-developed model.
- **Topics:** PRESELECT (LLM quality train data selection), PyTorch vs Tensorflow, Analyzing ECB communications improves forecasting of interest rate decisions, UQLM (Uncertainty Quantification for Language Models), Trading Framework: Function Calling. (More topics to be added as I have several research papers lined up for reading)
- **Key Learnings:** Explore advanced and emerging topics in financial NLP, including data selection for LLMs and the application of function calling in trading frameworks.

### Week 12/13: Project 2 (Final Project)
- **Description:** This project involves developing a backend project throughout the course. At the end, the best classification technique will be wrapped in an MCP or function calling framework for paper trading.
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