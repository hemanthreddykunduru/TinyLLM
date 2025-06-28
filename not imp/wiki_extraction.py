import wikipedia
import os

wikipedia.set_lang("en")
output_dir = "wiki_ai_articles"
os.makedirs(output_dir, exist_ok=True)
topics = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Natural language processing",
    "Retrieval-augmented generation",
    "Generative artificial intelligence",
    "Transformers (machine learning model)",
    "Computer vision",
    "Neural network",
    "ChatGPT",
    "GPT-4",
    "BERT (language model)",
    "Large language model",
    "Reinforcement learning",
    "Supervised learning",
    "Unsupervised learning",
    "Transfer learning",
    "Prompt engineering",
    "Word embeddings",
    "Vector database",
    "LangChain"
]

# To store related articles from links (optional expansion)
extra_topics = set()

# Download each main article
for idx, title in enumerate(topics, 1):
    try:
        print(f"[{idx}] Downloading: {title}")
        page = wikipedia.page(title, auto_suggest=False)
        content = page.content

        # Safe filename
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:100]
        filepath = os.path.join(output_dir, f"{safe_title}.txt")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # Collect related links for optional expansion
        links = page.links[:10]  # Limit to top 10 links per page
        extra_topics.update(links)

    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError, Exception) as e:
        print(f"❌ Skipped '{title}': {e}")

print("\n✅ Main topic download complete.")

# Optional: Download related linked articles (comment out if not needed)
print("\n🔁 Downloading related linked articles...")
for i, title in enumerate(extra_topics, 1):
    try:
        print(f"   - [{i}] {title}")
        page = wikipedia.page(title, auto_suggest=False)
        content = page.content

        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:100]
        filepath = os.path.join(output_dir, f"{safe_title}.txt")

        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

    except:
        continue

print("\n📁 All articles saved in:", output_dir)
