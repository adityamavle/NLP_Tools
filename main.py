from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#from transformers import pipeline
import joblib
#import numpy as np
#import pandas as pd
import os
import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="Keyword Extractor API",
    description="A simple API that use an NLP Model to extract keywords out of word",
    version="0.1",
)
@app.get("/extract-keywords")
def keywordextract(text):
        n_gram_range = (1, 1)
        stop_words = "english"
        count = CountVectorizer(ngram_range=n_gram_range,stop_words=stop_words).fit([text])

        candidates = count.get_feature_names()

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        doc_embedding = model.encode([text])
        candidate_embeddings = model.encode(candidates)

        top_n =5
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0]][-top_n:]

        print(keywords)

        print(type(keywords))

        s = ",".join(keywords)

        print(s)
        return s
    # keywrdextract(data)
if __name__ == "__main__":
     text="""The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU
[16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building
block, computing hidden representations in parallel for all input and output positions. In these models,
the number of operations required to relate signals from two arbitrary input or output positions grows
in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes
it more difficult to learn dependencies between distant positions [12]. In the Transformer this is
reduced to a constant number of operations, albeit at the cost of reduced effective resolution due
to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as
described in section 3.2.
Self-attention, sometimes called intra-attention is an attention mechanism relating different positions
of a single sequence in order to compute a representation of the sequence. Self-attention has been
used successfully in a variety of tasks including reading comprehension, abstractive summarization,
textual entailment and learning task-independent sentence representations [4, 27, 28, 22].
End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned
recurrence and have been shown to perform well on simple-language question answering and
language modeling tasks [34].
To the best of our knowledge, however, the Transformer is the first transduction model relying
entirely on self-attention to compute representations of its input and output without using sequencealigned
RNNs or convolution. In the following sections, we will describe the Transformer, motivate
self-attention and discuss its advantages over models such as [17, 18] and [9].
3 Model Architecture
Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35].
Here, the encoder maps an input sequence of symbol representations (x1; :::; xn) to a sequence
of continuous representations z = (z1; :::; zn). Given z, the decoder then generates an output
sequence (y1; :::; ym) of symbols one element at a time. At each step the model is auto-regressive
[10], consuming the previously generated symbols as additional input when generating the next.
The Transformer follows this overall architecture using stacked self-attention and point-wise, fully
connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1,
respectively."""
     #c=keybertextracter(text)
     keywordextract(text)
