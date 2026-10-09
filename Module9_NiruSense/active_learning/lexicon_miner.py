from collections import Counter
from transformers import AutoTokenizer
import re


class LexiconMiner:
    def __init__(self, model_name="Davlan/afriberta-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Basic English/Swahili stopwords (mock list for now)
        self.stopwords = set(
            ["the", "is", "and", "na", "ya", "wa", "kwa", "ni", "za", "la"]
        )

    def mine_unknowns(self, texts, top_k=50):
        """
        Finds frequent tokens/words in the texts.
        Since we use a subword tokenizer, 'unknown' is relative.
        We look for words that might be split weirdly or just frequent words in uncertain texts.
        Alternatively, we can just do simple whitespace tokenization and count words
        that are NOT in a known dictionary (if we had one).

        For this implementation, we'll count word frequencies in the uncertain text
        to highlight potential slang terms.
        """
        word_counts = Counter()

        for text in texts:
            # Simple cleaning
            text = re.sub(r"[^\w\s]", "", text.lower())
            words = text.split()

            for word in words:
                if word not in self.stopwords and len(word) > 2:
                    word_counts[word] += 1

        return word_counts.most_common(top_k)


if __name__ == "__main__":
    miner = LexiconMiner()
    texts = ["Hii ni sheng mpya manze", "Manze ni noma", "Kwani ni kesho"]
    print(miner.mine_unknowns(texts))
