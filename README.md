# Financial NLP for Trading: Hands-on Course
__Author__ = 'Firas Obeid'

## Syllabus: NLP in Finance Hands-on Course (13 Weeks)

### Course Overview:

The scope of the course is focused on language models and NLP techniques for event driven trading, but several techniques can be generalized to other disciplines as well. We will re-visit traditional NLP techniques up to the most recent advancements in language modeling. Throughout the course we will use a news headlines dataset, collected from Refinitiv’s (Reuters) API, as we go through various language modeling approaches. The course focuses on how classification, largely, is the fore front of NLP models output mechanism, and the lessons will be adapted to a research paper that the lecturer replicated and developed a language model (LM) around back in 2020. Then, we will further challenge that LM in classifying news headlines Buys/Sells with the various deep learning models starting with simple rules (heuristics) to tuning an off the shelf LLM, finally present new techniques while classification manifests as the core paradigm
_______________________________________________________________
### Week 1: Survey of traditional NLP techniques

**Scope:** Build intuition behind rule-based NLP systems and traditional text processing pipelines.

**Technologies**:
1. Textblob
2. NLTK
3. BoW
4. TF-IDF
5. Sentiment Analysis
6. n-grams
7. Hugging Face Dataset

**Session Plan**:
- This lecture sets the tone and scope for the rest of the course by presenting a self collected news headlines dataset from Refinitiv
- Use the financial news headlines dataset to walk through NL"Processing" traditional techniques
- Build a simple lexicon based sentiment model after defining our target variable
- Introduce Hugging Face datsets

*HW*: Analyze the news headlines dataset by cleaning it up & building a lexicon-based sentiment model.

### Week 2: Topic Modelling

**Scope:** Get familiar with Unsupervised Learning techniques for learning latent topics.

**Technologies:**
1. Probabilistic Algorithm: LDA
2. Deterministic Algorithm: NME
3. Segmentation

**Session Plan**:
- Extract topic latent topic categories from the news healines dataset
- Explain a probablistic algorithm for latent topic modelling
- Explain a matric factorization algorithm for topic modelling
- Based on most important tokens, label the news healines categories for segmentinf future model performance in each category

*HW*: Analyze the news headlines dataset to learn latent topics and label headlines accordingly.


### Week 3: Text Representation (Tokenization & Embedding)

**Scope:** Walk through different tokenization techniques & explain evolution of embeddings.

**Technologies**:
1. Word2Vec
2. Doc2Vec
3. n-gram tokens
4. Embeddings
5. Tokenization
6. BPE

**Session Plan**:
- Demonstrate several tokenization techniques.
- Walk through advanced text processing & strategies with n-grams tokens.
- Explain evolution of embeddings vectors from Word2Vec to Doc2Vec and recent embeddings models.
- Explain how to use N-dimensional embeddings by averaging them or normalizing them.

*Formula:*
```
emb = emb / √Σ(emb)²  (L2 normalization)
```
*NOTE*: Start searching for a research paper not older than 3 years, read it, own it and live it!


### Week 4: Develop your First Language Model

**Scope**: Develop intuition behind LSTM architectures for Language modeling for learning text (news headlines) representation & text generation.

**Technologies**:
1. LSTM
2. Language Models
3. Embeddings
4. Text Representation
5. Text Generation
6. Semi-supervised Learning

**Session Plan**:
- Explain difference between supervised / semi-supervised / weak/self-supervised.
- Develop an LSTM model for training news headlines embeddings on character (byte) level to token.
- Develop an LSTM model for text generation.
- Explain the notion of LSTMs for representation vs generation.


### Week 5: Project 1

**Descritption** : Build a story from HW1 & HW2 through analyzing the news headlines tokens, topics, token stats, news headline stats and so on. Consider this as a warm up for your final project.


### Week 6: Fine tune the LSTM-LM for News Headlines Classification  

**Scope**: Pickup week 4’s Language model and fine-tune that on downstream classification task for news headlines buy/sell predictions.  

**Technologies**:  
1. Transfer Learning  
2. LSTM  
3. Fine Tuning  
4. Hugging Face (Spaces)  
5. Gradio Deplyments

**Session Plan**:  
- Fine tune our LM on a classification task  
- Capture model performance using targeted metric (business metric)  
- Explain random vs fixed sampling from the original LM that was developed  
- Demo HF deployment of the event-driven LM for trading.  


### Week 7/8: Alternative NLP Classification techniques  

**Scope**: Survey several modern classification techniques for broader exposure and creating contender (competing) models with the previously developed LSTM classifier.  

**Technologies**:  
1. CBOW  
2. Skip-gram  
3. FastText  
4. In-context learning  
5. Skrub (Load in Embeddings) 
6. NLI (inference)  
7. Zero-shot Classifiers (CrossEncoder Transformers)
8. Embedding Models  (BERTs)
9. Cosine similarity with SVC  

**Session Plan**:  
- The session will span over two lectures to cover various NLP & NLI classification techniques.  
- Develop intuition behind all techniques  
- Develop models and compare with the LSTM LM, trained on news headlines classification.  
- Develop intuition behind similarity measures for classification as well.  
- Discuss how natural language inference works.  

### Week 9: Survey encoder, decoder, encoder-decoder LLMs  
**Scope**: Explain various flavours of encoders are there and how each of the three LLM paradigms work from input/output perspective.  

**Technologies**:  
1. Encoders  
2. Decoders  
3. MLM  
4. Self-Supervised Learning  
5. BERT  
6. Cross-Encoders  
7. Hugging Face model loaders  
8. Attention  

**Session Plan**:  
- Explain how encoders, decoders or encoders-decoders work, specifically input/output and high-level inner components.  
- Show how self-supervision manifests in encoder training.  
- Explain different encoder training paradigms.  
- Demo how to use several model instances from HF.

### Week 10: Fine-Tune on LLM for news headline classification  
**Scope**: Further expand our classification leaderboard by fine-tuning an open-source LLM and compare results with previous classifiers / language models.  

**Technologies**:  
1. Fine-tuning  
2. PEFT  
3. Adapters  
4. QLoRA  
5. Hugging Face Trainer  

**Session Plan**:    
- Build intuition off using LLMs to fine-tune.  
- Discuss the paradigm ogg in context learning to finetuning on the other end of the scale
- Dicuss QLora and any attention mechansism variant
- Demo fine-tuning Google's Gemma 3 LLM for classifying the news headlines dataset worked with throughout the whole course.

### Week 11: Special Topics  
**Scope**: Discuss special selected topics from recent research papers to trading agentic approach from best model developed.  

**Topics**:  
1. PRESELECT: LLM quality train data selection  
2. Gemini or Gemma white paper 
3. LLM Function Calling : Wrapping Up our Event Driven Trading Framework

**Session Plan**:  
- Explain a strategy for train data selection for LLMs.  
- Explain Google’s white paper.  
- Function Calling will be the creme of this session. I will demonstrate how to use Gemini or Gemma (Google LLM) to retrieve news headlines for stocks that have the highest VWAP ,using an external API to the LLM, the same day the prediction will be made.  

### Week 12/13: Project 2 (Final Project)
- *Presentations*: 15pts
- *Deployment*: 15pts
- *Github Documentation*: 10pts
- *Research Paper Replication*: +10pts
- Execution (Ideal/Code): 50pts
- Peer Review 20% of Project grade (TBD)
*Note*: If project 2 is based on a published research paper, you will automatically get +10pts and waive the separate research paper assignment.

### Grading Breakdown:
- *HW*: 20%
- *Project 1*: 30%
- *Project 2*: 40%
- *Research Paper Results Replication*: 10%

*Note*: Project 2 & the research paper can be combined.

### Code Scope:  
- Build hands-on intuition behind text processing & fundamental NLP libraries such as NLTK, gensim, spacy.  
- Build several tokenizers and encoding schemes.  
- Host training sets & on HF along with validation sets.  
- Introduce several HF LLM pipelines, mainly, classification & embedding related.  
- Develop an LM locally and push it to HF models for usage through the SDK "transformers."  
- Deploy locally developed LM on HF spaces as a Gradio app.  
- Show same LM TensorFlow version in PyTorch.  
- Demo LLM fine-tuning through the "Trainer" API and push that to HF transformers after that.  
- Reveal a framework using fuction calling for event driven trading based on price movment heuritcs filtering rules.


### Future Directions:

### Additional Resources
* [AI Hedge Fund Repository](https://github.com/virattt/ai-hedge-fund)
    The hedge fund consists of six agents:
    1. **Market Data Agent**: Gathers market data, including stock prices and fundamentals.
    2. **Quant Agent**: Calculates signals, such as MACD, RSI, and Bollinger Bands.
    3. **Fundamentals Agent**: Analyzes profitability, growth, financial health, and valuation.
    4. **Sentiment Agent**: Examines insider trades to determine insider sentiment.
    5. **Risk Manager**: Determines risk metrics, including volatility and drawdown.
    6. **Portfolio Manager**: Makes final trading decisions and generates orders.
* [Gemini API Documentation](https://ai.google.dev/gemini-api/docs/models/gemini#text-embedding)
* [Fine-Tune Gemma using Hugging Face Transformers and QloRA ](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)
* [Data Science Agent](https://developers.googleblog.com/en/data-science-agent-in-colab-with-gemini/)