# Conversational Podcast-driven MultiModal Dataset, and Automated Method for its Extraction from Natural Data 

## Motivation and data description 

Factual information retrieval (FIR) systems play a critical role in conversational dialogue systems, as they help ground responses in factual information from a knowledge base, thereby reducing the risk of hallucinations. Text-based FIR systems are typically trained evaluated on large-scale datasets of question-answer pairs, such as MSMARCO (TODO ref), TriviaQA (TODO ref), and Natural Questions (TODO ref). 

While text-based IR is extensively studied, it is also important to consider multimodal FIR systems, which can use both text and audio information to retrieve relevant facts for users' information needs. Moreover, it is essential to consider natural speech queries that simulate real-world interactions as reallistically as possible. 

To this end, we build a conversational FIR dataset based on the Spotify podcasts dataset (TODO ref). This dataset contains approximately 100,000 episodes with 60,000 hours of audio content, and can provide a rich source of spontaneous speech that closely resembles natural queries in conversational settings.

The dataset we build consiers both the original audio recordings and their corresponding transcriptions, enabling the development and evaluation of retrieval systems across four modalities of search: speech queries to speech answers, speech queries to text answers, text queries to speech answers, and text queries to text answers. This comprehensive approach facilitates the training and evaluation of retrieval systems that leverage large language models (LLMs) adapted to speech via a shared representation space, as outlined in D2.1.

In comparison to existing speech-based FIR datasets like TODO ref, our datasets relies exclusively on completely natural questions and answers extracted from authentic conversations, rather than relying on synthetic or crowdsourced data. This aims at building a dataset that is more representative of real-world conversational interactions, and can be used to train and evaluate FIR systems that are robust to the variability and noise present in spontaneous speech. Moreover, our dataset is different from the 2020/2021 TREC Podcasts Track also based on the Spotify dataset (TODO ref), because the latter contains only textual queries formulated by the track organizers, which may not fully capture the nature of natural speech queries.


## Architecture of the question/answer selecting pipeline 

Conventional IR datasets usually consist of three primary components: a collection of documents (the units to be retriebed), a set of queries (that represent the informations needs of users), and relevance judgments (that indicate which documents are relevant to which queries). The task for text-based FIR systems is to return a ranked list of documents for each query, based on their estimated relevance to the query.

Our goal is to extend this framework by constructing an IR dataset that incorporates both spoken and textual queries and documents derived from the Spotify podcast dataset. From the podcast dataset, we aim to identify suitable questions that simulate user information needs in a dialogue context, along with asnwers that serve as relevant passages from a large speech database. Specifically, we seek to extract:

- High-quality questions that are:
    - Self-contained: unambiguous for someone who hasn't heard the rest of the conversation
    - Information-seeking: asks for general factual knowledge, not personal experiences of the speakers

- High-quality answers that are:
    - Self-contained: providing complete information without requiring additional context
    - Relevant: directly addressing the corresponding question

To extract these suitable question-answer pairs, we developed a processing pipeline using 30 random podcasts as a development set. The final automated pipeline consists of the following stages:

1. Podcast transcription: we converted each audio podcast to text transcriptions using Whisper Large-v3 (TODO ref). The output is running text, not containing speaker information or timestamps.
2. Question extraction: For each transcript, we extracted at most one question using GPT-4o-mini (TODO ref) with instructions to identify self-contained, information-seeking questions. For each podcast, we extracted at most one question i.e., no question is extracted if the model does not identify a suitable question.
3. Question hallucination filter: since the question extraction model may hallucinate questions that do not appear in the transcript, we filter out questions with less than 90% string matching with the input transcript to ensure the questions actually appear in the original content.
4. Question quality filter: because the question extraction model may not always produce high-quality questions, we additionally evaluate each extracted question using GPT-4o-mini with instructions to determine if they are both self-contained and information-seeking, producing boolean values for each criterion.
5. Answer extraction: for each validated question, we located the corresponding answer within the transcript using GPT-4o-mini.
6. Answer filter: we tag as "invalid" answers that appear before their corresponding questions in the transcript, that have less than 90% string matching with the input content, or that are extremely long (over 500 words), and discard them.

The prompts used throughout this pipeline were developed through lightweight prompt engineering on the development podcast set.

Preliminary results on an independent set of 200 random podcasts showed that the pipeline was able to extract question-answer pairs from 10% of the podcasts (20 out of 200), with a precision of 50% (10 out of 20 were identified as high-quality by manual inspection). 

Based on these preliminary results, we decided to run the pipeline on just a random subset of TODO podcasts, instead of the whole dataset, because it should be enough to provide a reasonable amount of queries at a reasonable computational cost. Additionally, we decided to add a final manual filtering step to ensure the quality of the dataset, which consists on annotating four criteria for each question-answer pair: question self-containment, question information-seeking nature, answer self-containment, and answer relevance. We only keep samples that meet all four criteria in the final dataset.

Using the final set of validated question-answer pairs, we built a conventional text IR dataset with the following components:

* Queries: the extracted questions
* Documents: passages of 100 words with 50-word overlap between each other, extracted from the podcasts' transcripts. Importantly, we remove the queries strings within these passages to prevent trivial matching.
* Relevance judgments: established based on word overlap between the answers and the document passages. For example, if a passage contains the full answer, it is marked with a relevance of 1.0; if it contains half of the answer words, it is marked with a relevance of 0.5; and so on.

The resulting IR task requires systems to identify relevant passages for a given question within the entire database of passages across all podcasts.

The final dataset consists of TODO queries, TODO documents, with an average of TODO relevant documents per query. Examples of question-answer pairs extracted by the pipeline are shown in Table 1.

To evaluate retrieval performance, we consider use two standard metrics, Recall@k and NDCG@k, which for a given query are defined as:

$$
\begin{align*}
\text{Recall@k} & = \frac{\text{hits@k}}{\min(\text{R}, k)} \\
\text{NDCG@k} & = \frac{\text{DCG@k}}{\text{IDCG@k}}
\end{align*}
$$

where hits@k is the number of relevant documents retrieved in the top k results, R is the total number of relevant documents for the query, DCG@k is the discounted cumulative gain at k, and IDCG@k is the ideal discounted cumulative gain at k. DCG@k is defined as:

$$
\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$

where $rel_i$ is the relevance of the document at rank i, and IDCG@k is the DCG@k when the results are sorted by relevance, i.e., the ideal ranking.

To measure performance in the complete dataset, both metrics are averaged across all queries. 
