To better analyse the evolution of artificial intelligence research, we have established a two-dimensional analytical framework based on the following axes.

## Temporal Evolution

We used a corpus covering the period from 2007 to 2021:

* **2007-2011**: Traditional machine-learning-dominated.
* **2012-2016**: The Golden Age of Deep Learning and Neural Networks.
* **2017-2021**: The Rise of the Transformer, and An Introduction to Large Language Models.

These axes are not random yearly splits. They link to key changes in AI history, like the AlexNet breakthrough in 2012 and the Transformer structure launched in 2017.

## The depth of semantic analysis

We compared two fundamentally different text representation methods: lexical representation using TF-IDF and deep contextual representation using SBERT. This axes allows us to investigate whether developments in the field have been driven merely by the introduction of new terminology (captured by TF-IDF) or by a fundamental shift in the underlying meaning and context of research topics (captured by SBERT).

## My Hypothesis

We hypothesise that lexical methods (TF-IDF/LDA) will be highly sensitive to the emergence and decline of specific terms, with time drift rising sharply when new technologies are invented. Conversely, we hypothesise that semantic embeddings (SBERT) will exhibit smoother, more continuous trajectories in the PCA-reduced space, reflecting broader paradigm shifts in which concepts naturally intermingle over time.



## Data Processing (Text Preprocessing \& Period Splitting)



Before applying our text representation models, we constructed a robust data pipeline to clean and standardise the corpus:

* **Corpus Loading \& Splitting**: We loaded the dataset and partitioned it into three distinct temporal periods (2007-2011, 2012-2016, 2017-2021) based on the publication year.

* **Tokenisation \& Cleaning**: The abstract of each paper was converted to lowercase. We removed punctuation, numbers, and standard English stop words to reduce noise.

* **Standardisation**: We applied lemmatisation to ensure different forms of the same word (e.g., 'networks' and 'network') were treated identically.



\## Individual Method: Lexical Representation (TF-IDF)



As part of the comparative analysis, my focus was on constructing the lexical baseline using TF-IDF (Term Frequency-Inverse Document Frequency).



\* \*\*What it is\*\*: TF-IDF is a statistical measure that evaluates how relevant a word is to a document within a collection. It assigns higher weights to terms that appear frequently in a specific abstract but rarely across the entire corpus.

\* \*\*Why we used it\*\*: While SBERT captures deep contextual meaning, it acts as a black box. TF-IDF is highly interpretable. We used it specifically to track \*\*Terminology Drift\*\* — identifying exactly which unique jargon (e.g., 'Transformer', 'CNN') emerged or faded in specific eras.

\* \*\*How it was implemented (Axis 1 \& 2)\*\*:

&#x20; 1. \*\*Vectorisation\*\*: I applied `TfidfVectorizer` to convert the cleaned abstracts into a sparse matrix of n-gram features.

&#x20; 2. \*\*Axis 1 (Distance)\*\*: I computed the mean TF-IDF vector (centroid) for each of the three time periods and calculated the cosine distances between consecutive periods to measure the shift in academic vocabulary.

&#x20; 3. \*\*Axis 2 (Classifier)\*\*: To prove which words actively drive the temporal boundaries, I trained a \*\*Logistic Regression classifier\*\* on the TF-IDF matrix. By analysing the highest positive coefficients, I successfully extracted the "discriminative features" (top 15 exclusive keywords) that strictly define each era.

