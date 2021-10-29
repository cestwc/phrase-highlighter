from kitanaqa.augment.term_replacement import DropTerms, RepeatTerms
drop_word_sents = DropTerms()
repeat_word_sents = RepeatTerms()

import random

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data

from nltk.tokenize.treebank import TreebankWordDetokenizer
word_detokenize = TreebankWordDetokenizer().detokenize


def highlight_on_words(article_original, highlights_original):
	W = set(highlights_original).intersection(set(article_original))
	ignore_mask = list(map(lambda x: int(x in W), article_original))

	# labels = [int(bool(h or not a)) for h, a in zip(ignore_mask, attention_mask)]
	labels = [0] * len(article_original)

	# get spans
	starts = []
	ends = []
	running = False
	for i, v in enumerate(ignore_mask):
		if v == 1 and not running:
			starts.append(i)
			running = True
		elif v == 0 and running:
			ends.append(i - 1)
			running = False
	if running:
		ends.append(i)
		running = False

	spans = [e - s + 1 for s, e in zip(starts, ends)]

	while W:
		# get longest segment
		segment = spans.index(max(spans))
		k = starts[segment]
		while k < ends[segment] + 1:
			labels[k] = 1
			ignore_mask[k] = 0
			W.discard(article_original[k])
			k += 1

		# recalculate spans
		spans = [0] * len(spans)
		for i, (s, e) in enumerate(zip(starts, ends)):
			for v in article_original[s:e+1]:
				if v in W:
					spans[i] += 1


	return labels, ignore_mask

def interpose(words, labels, ignore_mask, sos = 'ORANGEGROUPSTART', eos = 'ORANGEGROUPEND', isos = 'ORANGEIGNORESTART'):
	labeled = []
	running = False
	for i, (w, s) in enumerate(zip(words, labels)):
		if ignore_mask[i] == 1 and not running:
			labeled.append(isos)
			labeled.append(w)
			running = True
		elif ignore_mask[i] == 0 and running:
			labeled.append(eos)            
			labeled.append(w)
			running = False            
		elif s == 1 and not running:
			labeled.append(sos)
			labeled.append(w)
			running = True
		elif s == 0 and running:
			labeled.append(eos)            
			labeled.append(w)
			running = False
		else:
			labeled.append(w)

	if running:
		labeled.append(eos)
		running = False

	return labeled


def syn_rep(words):

	# Identify the parts of speech
	tagged = nltk.pos_tag(words)

	for i in range(0, len(words)):
		if random.random() > 0.5:
			continue

		replacements = []

		# Only replace nouns with nouns, vowels with vowels etc.
		for syn in wordnet.synsets(words[i]):

			# Do not attempt to replace proper nouns or determiners
			if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT':
				break

			# The tokenizer returns strings like NNP, VBP etc
			# but the wordnet synonyms has tags like .n.
			# So we extract the first character from NNP ie n
			# then we check if the dictionary word has a .n. or not 
			word_type = tagged[i][1][0].lower()
			if syn.name().find("."+word_type+"."):
				# extract the word only
				r = syn.name()[0:syn.name().find(".")]
				replacements.append(r)

		if len(replacements) > 0:
			# Choose a random replacement
			replacement = replacements[randint(0,len(replacements)-1)]
			words[i] = replacement

	return ' '.join(words)

def augment(text):
	text = syn_rep(text)
	texts = drop_word_sents.drop_terms(text, num_terms=1, num_output_sents=1)
	if len(texts) > 0:
		text = texts[0]
	texts = repeat_word_sents.repeat_terms(text, num_terms=1, num_output_sents=1)
	if len(texts) > 0:
		text = texts[0]
	return text

def tokenSubstitute(text):
	text = text.replace('ORANGEGROUPSTART', '<s>')
	text = text.replace('ORANGEGROUPEND', '</s>')
	text = text.replace('ORANGEIGNORESTART', '<mask>')
	return text

def preprocess(article, highlights, aug = False):
	article_original = word_tokenize(article)
	highlights_original = word_tokenize(highlights)
	labels_original, ignore_mask = highlight_on_words(article_original, highlights_original)
	labeled_article = interpose(article_original, labels_original, ignore_mask) # list
	if aug and random.random() > 0.5:
		labeled_article = augment(labeled_article) # string
	else:
		labeled_article = word_detokenize(labeled_article) # string
	labeled_article = tokenSubstitute(labeled_article) # string
	return labeled_article.lower()

def mark(labeled_article):

	article_ids = [0]
	labels = [1]
	ignoring = False
	highlighting = False
	for i, v in enumerate(labeled_article):
		if i == 0:
			pass
		elif i == 511:
			article_ids.append(2)
			labels.append(1)
			break
		elif v == 0:
			highlighting = True
			ignoring = False
		elif v == 50264:
			ignoring = True
			highlighting = False
		elif v == 2:
			highlighting = False
			ignoring = False
		else:
			article_ids.append(v)
			if highlighting:
				labels.append(1)
			elif ignoring:
				labels.append(-100)
			else:
				labels.append(0)

	assert len(article_ids) == len(labels)
	return article_ids, labels
