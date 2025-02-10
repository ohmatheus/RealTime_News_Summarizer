# Work In Progress

# RealTime_News_Summarizer
Using DeepSeek + Langchain for realtime RAG summarizing news from selected web site(s) with API calls and scrapping.

### About This Project
Just a tiny project to develop my skills with langchain, scrapping, RAG, and usage of deepseek.
The idea is to have an app prompting user for news on a specific topic on the last 24h (may be a variable). Using langchain for RAG and automation. scrapping (beautifulsoup4), and deepseek for summerizing. 
As i am french, my goal is to make this work with one of the major news french website : [Le Monde](https://www.lemonde.fr/).

For the sake of this project, this will stay a POC and I would use all the least resource-intensive methods. Namely the lightest distilled model from deepseek : [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) using Qwen-1.5B architecture finetuned on data generated by DeepSeek R1 showing fine results. Should be enought for summarization. We will talk about possible improvements at the end of the project.

### No link to application
I have not 

# RAG System (simple)


## 1. Scrapping 'Le Monde'

## 2. Vector Database

Using General Text Embeddings (GTE) model ("thenlper/gte-large") as text embbeder via `HuggingFaceEmbeddings (langchain)`. It seemed to me to be a good compromise between performance and execution speed.

Using facebook's FAISS as Nearest Neighbor search algorithm with cosine similarity. 


## 3. Reader - LLM
Deepseek  
`<think></think>` tokens
prompt


"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" quantized 4-bit

## 4. Results

## 5. Possible Improvements
- Better selection of articles to put in the context :
    - Reranking
    - Adding metadata (like tags) in documents.
- Better prompt : Currently asked to iterate throught each document -> leading to hallucianation because he is sometimes linking documents that have no link whatsoever with our subject's.
- Test with better LLM. Maybe one with better french capabilities so i can make an all french application.
- Adding links of selected documents at the end.
- Narrow the scrapping and the search for only the last 24h news. Maybe find a way to pre-select only the most important new (internationnal for exemple)



## Links
### Inspirations:
- [Langchain (Upgraded) + DeepSeek-R1 + RAG Just Revolutionized AI Forever](https://pub.towardsai.net/langchain-upgraded-deepseek-r1-rag-just-revolutionized-ai-forever-27dcbb0e3493)
- [Developing RAG Systems with DeepSeek R1 & Ollama](https://sebastian-petrus.medium.com/developing-rag-systems-with-deepseek-r1-ollama-f2f561cfda97)
- (FR) [Scraper « le monde » et construire ton propre corpus](https://xiaoouwang.medium.com/scraper-le-monde-et-construire-ton-propre-corpus-d47fa81eb3d9)
### Worked with:
[DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)


```python
code dummy
```