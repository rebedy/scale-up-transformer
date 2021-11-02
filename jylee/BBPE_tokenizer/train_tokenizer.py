##### Train a tokenizer
# We choose to train a byte-level Byte-pair encoding tokenizer (the same as GPT-2), with the same special tokens as RoBERTa. Let’s arbitrarily pick its size to be 52,000.
# We recommend training a byte-level BPE (rather than let’s say, a WordPiece tokenizer like BERT) because it will start building its vocabulary from an alphabet of single bytes, 
# so all words will be decomposable into tokens (no more <unk> tokens!).
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("/home/edlab/wcshin/physionet.org/files/mimic-cxr-jpg/2.0.0/preprocessed_reports").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)

# Customize training
tokenizer.train(files=paths, vocab_size=15_000, min_frequency=2, special_tokens=[
    "[PAD]",
    "[SOS]",
    "[EOS]",
    "[SEP]",
    "[MASK]",
])

# Now let's save files to disk
tokenizer.save_model("./")
# We now have both a vocab.json, which is a list of the most frequent tokens ranked by frequency, and a merges.txt list of merges.
# What is great is that our tokenizer is optimized for Esperanto. Compared to a generic tokenizer trained for English, more native words are represented by a single, unsplit token.
# Diacritics, i.e. accented characters used in Esperanto – ĉ, ĝ, ĥ, ĵ, ŝ, and ŭ – are encoded natively. We also represent sequences in a more efficient manner. 
# Here on this corpus, the average length of encoded sequences is ~30% smaller as when using the pretrained GPT-2 tokenizer.