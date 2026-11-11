# Self-Attention Explained: How Transformers Understand Context 

 ## What Is Self-Attention?

Self-attention is a way for a model to compare each part of an input with every other part and decide what matters most. For each word, the model assigns attention weights to other words, giving more weight to those that help clarify its meaning.

For example, in “The dog chased the ball because it was excited,” self-attention can help connect “it” with “the dog.” By weighing these relationships across the sentence, the model builds a richer understanding of context than it would by considering each word in isolation.

## Why Context Matters

A word on its own can have more than one meaning. Consider **“bank”** in these two sentences:

- “She sat by the **bank** of the river.”
- “She deposited money at the **bank**.”

If we considered each word in isolation, “bank” would be ambiguous. The surrounding words—“river” in one sentence and “deposited money” in the other—help reveal which meaning is intended. Self-attention lets a transformer weigh these relationships, so a word’s representation reflects the context around it.

## Queries, Keys, and Values

For each token, a transformer creates three vectors: a **query**, a **key**, and a **value**. These are learned representations of the token, each used for a different purpose:

- A **query** represents what the token is looking for in the other tokens.
- A **key** represents what a token has to offer, helping determine whether it is relevant to a query.
- A **value** carries the information that can be passed along if the token is attended to.

The model compares a token’s query with the keys of other tokens. A stronger match produces a higher attention score. In practice, it calculates these scores with dot products, scales them, and applies softmax to turn them into weights that sum to one.

Those weights are then used to combine the tokens’ values. A token gets more information from tokens with higher weights, allowing its representation to reflect relevant context. For example, when interpreting *“it”* in a sentence, the query for *“it”* may match strongly with the key for the noun it refers to, so that noun’s value contributes more to the updated representation.

In short, **queries and keys determine where attention goes; values determine what information comes back**.

## The Self-Attention Calculation

Self-attention lets each word look at the other words in a sequence and decide which ones matter most for understanding it. The calculation follows three main steps:

1. **Compare words to find relevant context.** Each word is turned into three vectors: a **query**, a **key**, and a **value**. The query represents what the word is looking for; keys describe what other words offer. Comparing a word’s query with the keys of all the words produces similarity scores. A higher score means a word may be more relevant.

2. **Scale and normalize the scores.** The scores are scaled down to keep them from becoming too large, then passed through a softmax function. This turns them into attention weights that add up to 1. For example, a word might give most of its attention to one nearby word and less to the rest.

3. **Combine the value vectors.** Each word’s value is multiplied by its attention weight, and the results are added together. The resulting vector gives the word a context-aware representation.

So, when a transformer processes *“The animal didn’t cross the street because it was tired,”* the word *“it”* can give more attention to *“animal”* than to *“street.”* This helps the model use context to work out what *“it”* refers to.

## Multi-Head Attention and Transformer Context

A single attention operation gives each token one way to weigh information from the other tokens. **Multi-head attention** runs several attention operations in parallel, each with its own learned projections of queries, keys, and values. This lets the model build different views of the same sequence: one head might learn to connect a pronoun to a name, while another attends to nearby words or tracks a grammatical relationship. These patterns are learned rather than assigned in advance, and heads do not always have clearly distinct roles.

The outputs from all heads are combined and projected back into the model’s usual representation size. The result lets each token draw on multiple kinds of context at once.

In a Transformer block, self-attention is typically followed by a **feed-forward network**, which further processes each token’s representation. **Residual connections** help preserve the incoming information, while **layer normalization** helps keep the computations stable. In decoder blocks, self-attention is often masked so a token cannot attend to future tokens; encoder–decoder models may also use cross-attention to connect the decoder to the encoder’s output. Repeated across many blocks, these components let tokens refine their representations using increasingly rich context.

## Strengths, Limitations, and Applications

Self-attention helps a model connect information across an input, even when relevant words or features are far apart. In a sentence, for example, it can relate a pronoun to an earlier noun while also considering the surrounding context. Because attention is computed across many positions in parallel, it can also be more efficient to train than sequential approaches such as recurrent networks.

That flexibility comes with costs. Standard self-attention compares every position with every other position, so its compute and memory requirements grow roughly quadratically with input length. Long documents, high-resolution images, and long audio streams can therefore be expensive to process. Attention also relies on positional information to represent order, and its ability to use context is limited by the model’s input window. Finally, attention patterns can be useful to inspect, but they are not a complete or necessarily reliable explanation of a model’s decisions.

Self-attention is central to language models used for tasks such as writing, translation, summarization, and question answering. In computer vision, it helps models relate distant regions of an image, supporting classification, detection, and image generation. It also appears in speech recognition, audio and video understanding, multimodal systems that combine text with images or sound, and scientific applications such as protein modeling.
