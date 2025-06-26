import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
import base64

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt",encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings())


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
moods = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Soft(), css=".centered { display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 20px; } .centered img { width: 120px; opacity: 0.6; }") as dashboard:
    gr.Markdown("## <span style='color:#6C63FF'> What Should I Read Next?</span>")
    gr.Markdown("<span style='color:#6B7280'> Discover great book recommendations tailored to your taste and mood.</span>")

    with gr.Row():
        user_query = gr.Textbox(
            label="Tell us what you're in the mood for:",
            placeholder="e.g., a feel-good novel about friendship and travel"
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Pick a genre (optional):",
            value="All"
        )
        mood_dropdown = gr.Dropdown(
            choices=moods,
            label="What kind of vibe are you looking for?",
            value="All"
        )

    with gr.Row():
        submit_button = gr.Button("Find Books",variant="primary", elem_classes="medium-button", scale=0)

    gr.Markdown("### <span style='color:#8D83FF'> 🌟 Your Book Picks</span>")

    with gr.Column(visible=True,elem_classes="centered-placeholder") as placeholder_section:
        gr.Markdown("<span style='color:#9CA3AF'> Start by describing what kind of book you're in the mood for above.</span>")

    output = gr.Gallery(label="", columns=8, rows=2, visible = False)

    def wrapped_recommend_books(query, cat, tone):
        recs = recommend_books(query, cat, tone)
        return gr.update(visible=False), gr.update(value=recs, visible=True)


    submit_button.click(
        fn=wrapped_recommend_books,
        inputs=[user_query, category_dropdown, mood_dropdown],
        outputs=[placeholder_section, output]
    )

if __name__ == "__main__":
    dashboard.launch(share=True)