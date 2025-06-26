# What Should I Read Next?  

A smart book recommender system that understands what you're in the mood for and gives you curated suggestions based on genre and emotion - powered by LLMs, sentiment analysis, and semantic search.

## Demo

👉 [Try the App](https://huggingface.co/spaces/yukito96/book-recommender)

---

## Features

- **Semantic Search:** Matches your query to relevant book descriptions using OpenAI embeddings + Chroma vector store.
- **Emotion-Based Filtering:** Classifies books into 5 emotions - joy, sadness, anger, fear, surprise - using a fine-tuned HuggingFace DistilRoberta model.
- **Genre Filtering:** Pick from 20+ genres to narrow your recommendations.
- **Interactive UI:** Built with Gradio and deployed on HuggingFace Spaces.

---

## Project Structure

```├── app.py # Gradio app (main entry point)
├── books_with_emotions.csv # Final dataset with emotions
├── tagged_description.txt # Preprocessed descriptions for vector indexing
├── data-exploration.ipynb # Data cleaning and preprocessing
├── vector-search.ipynb # Embedding + vector DB creation (Chroma)
├── text-classification.ipynb # To harmonize the skewed distribution of Genre
├── sentiment-analysis.ipynb # Multilabel emotion classification
├── requirements.txt # Dependencies
```

---

## How It Works

1. **Cleaned Book Data**
   - Deduplicated, cleaned metadata.
   - Processed to retain relevant columns: `title`, `authors`, `thumbnail`, `description`, `isbn13`.

2. **Vector Indexing**
   - `tagged_description.txt` was created with book metadata.
   - Texts were embedded using `OpenAIEmbeddings` and stored in `Chroma`.
  
3. **Genre Balancing**
   - Analyzes and balances the skewed genre distribution in book data.
   - Prepares a harmonized dataset for more effective genre classification and downstream embedding.

5. **Emotion Tagging**
   - Used `DistilRoberta` to assign 5 emotion scores per book.
   - Saved final output to `books_with_emotions.csv`.
     
6. **Frontend**
   - Users input a query and optionally select a genre and emotion.
   - Backend performs:
     - `similarity_search(query)`
     - Optional category and tone filtering
   - Results shown with book covers + summary using Gradio Gallery.

---

## Deployment on Hugging Face Spaces

The app is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/yukito96/book-recommender) using:

---

## Installation (for local development)

```bash
git clone https://huggingface.co/spaces/yukito96/book-recommender
cd book-recommender
pip install -r requirements.txt
python app.py
