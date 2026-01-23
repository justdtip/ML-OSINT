How to get embeddings with Anthropic

While Anthropic does not offer its own embedding model, we have partnered with Voyage AI as our preferred provider for text embeddings. Voyage makes state of the art embedding models, and even offers models customized for specific industry domains such as finance and healthcare, and models that can be fine-tuned for your company.

To access Voyage embeddings, please first sign up on Voyage AI’s website, obtain an API key, and set the API key as an environment variable for convenience:

export VOYAGE_API_KEY="<your secret key>"
You can obtain the embeddings either using the official voyageai Python package or HTTP requests, as described below.

Voyage Python Package

The voyageai package can be installed using the following command:

pip install -U voyageai
Then, you can create a client object and start using it to embed your texts:

import voyageai

vo = voyageai.Client()
# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

texts = ["Sample text 1", "Sample text 2"]

result = vo.embed(texts, model="voyage-2", input_type="document")
print(result.embeddings[0])
print(result.embeddings[1])
result.embeddings will be a list of two embedding vectors, each containing 1024 floating-point numbers. After running the above code, the two embeddings will be printed on the screen:

[0.02012746, 0.01957859, ...]  # embedding for "Sample text 1"
[0.01429677, 0.03077182, ...]  # embedding for "Sample text 2"
When creating the embeddings, you may specify a few other arguments to the embed() function. Here is the specification:

voyageai.Client.embed(texts : List[str], model : str = "voyage-2", input_type : Optional[str] = None, truncation : Optional[bool] = None)
texts (List[str]) - A list of texts as a list of strings, such as ["I like cats", "I also like dogs"]. Currently, the maximum length of the list is 128, and total number of tokens in the list is at most 320K for voyage-2 and 120K for voyage-code-2.

model (str) - Name of the model. Recommended options: voyage-2 (default), voyage-code-2.

input_type (str, optional, defaults to None) - Type of the input text. Defalut to None. Other options:  query, document.

When the input_type is set to None, and the input text will be directly encoded by our embedding model. Alternatively, when the inputs are documents or queries, the users can specify input_type to be query or document, respectively. In such cases, Voyage will prepend a special prompt to input text and send the extended inputs to the embedding model.
For retrieval/search use cases, we recommend specifying this argument when encoding queries or documents to enhance retrieval quality. Embeddings generated with and without the input_type argument are compatible.
truncation (bool, optional, defaults to None) - Whether to truncate the input texts to fit within the context length.

If True, over-length input texts will be truncated to fit within the context length, before vectorized by the embedding model.
If False, an error will be raised if any given text exceeds the context length.
If not specified (defaults to None), Voyage will truncate the input text before sending it to the embedding model if it slightly exceeds the context window length. If it significantly exceeds the context window length, an error will be raised.