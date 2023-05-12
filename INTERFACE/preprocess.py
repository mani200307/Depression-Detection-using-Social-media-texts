from collections import Counter
from itertools import chain
import nltk
from nltk.util import ngrams
from nltk.util import bigrams
import nltk
import re
import ftfy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import contractions
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Cleaning process
def expandContracted(sentence):
    expanded_words = []
    for word in sentence.split():
        # using contractions.fix to expand the shortened words
        expanded_words.append(contractions.fix(word))
    res = ' '.join(expanded_words)
    return res

# LEMMATIZATION WITH POS TAG
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_func(text):
    lemmatizer = WordNetLemmatizer()
    # tokenize the sentence and find the POS tag for each token
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(text))

    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)

    return lemmatized_sentence

def Negation(sentence):	
  '''
  Input: Tokenized sentence (List of words)
  Output: Tokenized sentence with negation handled (List of words)
  '''
  temp = int(0)
  sentence = nltk.word_tokenize(sentence)
  for i in range(len(sentence)):
      if sentence[i-1] in ['not',"n't"]:
          antonyms = []
          for syn in wordnet.synsets(sentence[i]):
              syns = wordnet.synsets(sentence[i])
              w1 = syns[0].name()
              temp = 0
              for l in syn.lemmas():
                  if l.antonyms():
                      antonyms.append(l.antonyms()[0].name())
              max_dissimilarity = 0
              for ant in antonyms:
                  syns = wordnet.synsets(ant)
                  w2 = syns[0].name()
                  syns = wordnet.synsets(sentence[i])
                  w1 = syns[0].name()
                  word1 = wordnet.synset(w1)
                  word2 = wordnet.synset(w2)
                  if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
                      temp = 1 - word1.wup_similarity(word2)
                  if temp>max_dissimilarity:
                      max_dissimilarity = temp
                      antonym_max = ant
                      sentence[i] = antonym_max
                      sentence[i-1] = ''
  res=""
  while '' in sentence:
      sentence.remove('')
    
  for x in sentence:
    res += x + " "
  return res

# PRE-PROCESSING FUNCTION
def cleaning_process(a_tweet):
    res_tweet = ''
    # convert to lowercase
    a_tweet = a_tweet.lower()

    # if url links then don't append to avoid news articles
    # also check tweet length, save those > 5
    if re.match("(\w+:\/\/\S+)", a_tweet) == None and len(a_tweet) > 0:
        # remove @mention
        a_tweet = re.sub(r"(?:\@|https?\://)\S+", "", a_tweet)

        # expand contraction
        a_tweet = expandContracted(a_tweet)

        print("Before Neg:", a_tweet)

        a_tweet = Negation(a_tweet)

        print("After Neg:", a_tweet)

        # remove punctuation
        a_tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", a_tweet).split())

        # remove hashtag, @mention, HTML tags and image URLs
        a_tweet = ' '.join(re.sub(
            "(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<.>)|(pic\.twitter\.com\/.*)", " ", a_tweet).split())

        # remove numbers
        a_tweet = re.sub(r'\d+', '', a_tweet)

        # remove urls
        a_tweet = re.sub(r'https?://\S+|www\.\S+', '', a_tweet)

        # fix weirdly encoded texts (Unicode correction)
        a_tweet = ftfy.fix_text(a_tweet)

        # stop words
        stop_words = set(stopwords.words('english'))
        word_tokens = nltk.word_tokenize(a_tweet)
        a_tweet = [word for word in word_tokens if not word in stop_words or word == "not"]
        a_tweet = ' '.join(a_tweet)  # join words with a space in between them

        # lemmatization
        a_tweet = lemmatize_func(a_tweet)

        res_tweet = a_tweet
    return res_tweet

# To calculate bow for input text
def bow_for_line(corpus, top_words):
    bag_of_words = []
    for doc in corpus:
        unigram_words = doc
        bigram_words = list(bigrams(doc))
        trigram_words = list(ngrams(doc, 3))

        word_counts = Counter(doc)
        bigram_word_counts = Counter(bigram_words)
        trigram_word_counts = Counter(trigram_words)

        word_count = dict(word_counts)
        word_count.update(dict(bigram_word_counts))
        word_count.update(dict(trigram_word_counts))

        dict_words = dict(word_count)
        line_words = Counter(dict_words)

        row = [line_words[word] if word in line_words else 0 for word in top_words]
        bag_of_words.append(row)

    return bag_of_words

def bow(data_train):
  # Preprocess and tokenize the corpus
  corpus = [nltk.word_tokenize(doc.lower()) for doc in data_train]

  # Calculate the frequency of each word in the corpus
  word_counts = Counter(chain.from_iterable(corpus))

  bigram_words = []
  for doc in corpus:
    words = list(bigrams(doc))
    bigram_words.append(words)

  # Calculate the frequency of each bi and trigram words in the corpus
  bigram_word_counts = Counter(chain.from_iterable(bigram_words))

  trigram_words = []
  for doc in corpus:
    words = list(ngrams(doc, 3))
    trigram_words.append(words)

  trigram_word_counts = Counter(chain.from_iterable(trigram_words))

  word_count=dict(word_counts)
  word_count.update(dict(bigram_word_counts))
  word_count.update(dict(trigram_word_counts))

  word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
  dict_words = dict(word_count)
  final_words = Counter(dict_words)

  # # Select the top k most frequent words as columns
  top_words = [word for word, count in final_words.most_common(5000)]

  # # # Create the bag of words matrix
  bag_of_words = bow_for_line(corpus, top_words)
  
  return bag_of_words, top_words

