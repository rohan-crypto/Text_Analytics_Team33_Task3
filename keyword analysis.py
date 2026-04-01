#%%
import nltk
import numpy as np
import networkx as nx
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from collections import Counter
import joblib
from tqdm import tqdm

# Download the Punkt tokeniser models
nltk.download('punkt')
# Download part-of-speech (POS) tagging model
nltk.download('averaged_perceptron_tagger')

# Load the preprocessed data for keyword analysis
df = pd.read_pickle("data_preprocessed.pkl")

# Verify the number of rows
print("Number of rows:", df.shape[0])

# Verify the number of columns
print("Columns:", df.columns.tolist())

# Ensure that analysis is based on the abstract column
text_column = 'abstract'

# Regex for valid words
Valid_word = re.compile(r'^[a-zA-Z\-]+$')

#Calculate universal IDF to measure word importance for all of the abstracts
def calculate_universal_IDF(preprocessed_tokens_list):
    N = len(preprocessed_tokens_list)
    document_frequency = Counter()
    for tokens in tqdm(preprocessed_tokens_list, desc="IDF Progress"):
        unique_words = set(tokens)
        document_frequency.update(unique_words)
    idf = {w: np.log(N / (1 + f)) for w, f in document_frequency.items()}
    return idf
preprocessed_tokens_list = [str(text).split() for text in df[text_column].tolist()]
universal_IDF = calculate_universal_IDF(preprocessed_tokens_list)

#convert a sentence into a numerical vector using TF-IDF
def sentence_vector(tokens, vocabulary, universal_IDF):
    if not tokens:
        return np.zeros(len(vocabulary))
    vector = np.zeros(len(vocabulary))
    Term_frequency = Counter(tokens)
    for word, count in Term_frequency.items():
        if word in vocabulary:
            vector[vocabulary[word]] = (count / len(tokens)) * universal_IDF.get(word, 1.0)
    return vector

#Turn text it into sentences and a numerical vocabulary so that TextRank can calculate sentence importance
def textrank_summarise(preprocessed_text, top_n=3, position_bias=0.3):
    if not isinstance(preprocessed_text, str):
        return preprocessed_text
    sentences = sent_tokenize(preprocessed_text)
    if len(sentences) <= top_n:
        return preprocessed_text
    all_words = set(preprocessed_text.split())
    vocabulary = {w: i for i, w in enumerate(all_words)}

#Convert all sentences into normalised numerical vectors so they can be fairly based on their similarities
    vectors = np.array([sentence_vector(sent.split(), vocabulary, universal_IDF) for sent in sentences])
    #Normalise each sentence vector to have unit length
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    vectors = vectors / norm
    #Compute cosine similarity between all pairs of sentences
    similarity_matrix = np.clip(vectors @ vectors.T, 0, None)
    #Remove self-loops in the similarity graph
    np.fill_diagonal(similarity_matrix, 0)
    #Convert the similarity matrix into a graph using NetworkX
    Graph_sentences = nx.from_numpy_array(similarity_matrix)
    #Apply PageRank algorithm to the graph
    scores = nx.pagerank(Graph_sentences, max_iter=200, tol=1e-6)

    #Assign higher raw scores to sentences that appear earlier
    position_raw = np.exp(-0.5 * np.arange(len(sentences)))
    #Normalise the raw position scores so they sum to 1
    position_score = position_raw / position_raw.sum()
    #Combine TextRank score and position score into a single importance score for each sentence
    final_scores = {i: (1 - position_bias) * scores[i] + position_bias * position_score[i] for i in range(len(sentences))}
    #Pick the top important sentences based on the combined score
    ranked_index_row = sorted(final_scores, key=final_scores.get, reverse=True)[:top_n]
    #Return the selected sentences in the original order to form a readable summary
    return ' '.join(sentences[i] for i in sorted(ranked_index_row))

#use Penn Treebank tags to define the structure of a Noun phrase
Noun_Phrase_Grammar = r'KP: {(<JJ.*>|<NN.*>)*<NN.*>}'
Noun_Phrase_Parser = nltk.RegexpParser(Noun_Phrase_Grammar)

def extract_noun_phrases(text):
    #Split text into individual words
    tokens = word_tokenize(text)
    #Assign grammatical labels
    tagged = nltk.pos_tag(tokens)
    #turn text into structured tree
    tree = Noun_Phrase_Parser.parse(tagged)
#store extracted noun phrases
    phrases = []
    #Loop through noun phrase chunks
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'KP'):
        words = [w.lower() for w, _ in subtree.leaves()]
        valid_words = [w for w in words if Valid_word.match(w)]
        #filter phrase length
        if 1 <= len(valid_words) <= 5:
            #Join words into phrase
            phrase = ' '.join(valid_words)
            #Store phrase
            phrases.append(phrase)
    #output all the extracted noun phrases
    return phrases

def filter_redundant_phrases(ranked_phrases, top_n):
    filtered = []
    #Loop through ranked phrases
    for phrase in ranked_phrases:
        #check redundancy
        if not any(phrase in f or f in phrase for f in filtered):
            filtered.append(phrase)
        if len(filtered) >= top_n:
            break
    #Output final cleaned keyphrases
    return filtered

#extract and rank important keyphrases using TextRank graph
def textrank_keyphrases(text, top_n=10, window=5):
    if not isinstance(text, str):
        return []
    phrases = extract_noun_phrases(text)
    if len(phrases) < 3:
        return phrases[:top_n]
    unique_phrases = list(dict.fromkeys(phrases))
    Graph_sentences = nx.Graph()
    Graph_sentences.add_nodes_from(unique_phrases)
    for i, p in enumerate(phrases):
        for q in phrases[i+1:i+window]:
            if p != q:
                if Graph_sentences.has_edge(p, q):
                    Graph_sentences[p][q]['weight'] += 1
                else:
                    Graph_sentences.add_edge(p, q, weight=1)

    #Apply PageRank
    PageRank = nx.pagerank(Graph_sentences, weight='weight', max_iter=200, tol=1e-6)
    #Define final scoring function
    def final_score(phrase):
        return PageRank.get(phrase, 0) * (1 + 0.15 * (len(phrase.split()) - 1))
    #Rank phrases
    ranked = sorted(unique_phrases, key=final_score, reverse=True)
    #Remove overlapping phrases
    return filter_redundant_phrases(ranked, top_n)

# 4. Parallel Processing
print("\nRunning TextRank summarisation & keyphrase extraction...")

def process_summaries(texts):
    return [textrank_summarise(t) for t in texts]

def process_keyphrases(texts):
    return [textrank_keyphrases(t) for t in texts]

#Split the data into chunks for parallel processing
n_cores = joblib.cpu_count()
text_chunks = np.array_split(df[text_column].astype(str), n_cores)

#Run parallel summarisation
df['summary'] = sum( joblib.Parallel(n_jobs=n_cores)( joblib.delayed(process_summaries)(chunk) for chunk in tqdm(text_chunks, desc="Summarising") ), [] )

#Run parallel keyphrase extraction
df['keyphrases'] = sum( joblib.Parallel(n_jobs=n_cores)( joblib.delayed(process_keyphrases)(chunk) for chunk in tqdm(text_chunks, desc="Extracting Keyphrases") ), [] )

#Displaying the summary and keyphrases for the first 3 rows of abstracts
pd.set_option('display.max_colwidth', 120)
print("\n"+"="*70)
print("First 3 rows")
print("="*70)
for index_row, row in df[[text_column,'summary','keyphrases']].head(3).iterrows():
    print(f"\n[Row {index_row}]")
    print(f"Summary: {str(row['summary'])[:200]}...")
    print(f"Keyphrases: {', '.join(row['keyphrases'])}")
df.to_pickle("data_with_nlp.pkl")
df[[text_column, 'summary', 'keyphrases']].to_csv("nlp_results.csv", index=False)
print("\nSaved > data_with_nlp.pkl, nlp_results.csv")