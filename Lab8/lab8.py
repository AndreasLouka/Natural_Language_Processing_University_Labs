import nltk
from nltk.corpus import brown
brown_words = brown.words()
unigram_count = nltk.FreqDist(brown_words)
bigram_count = nltk.FreqDist(nltk.bigrams(brown_words))

with open("questions.txt") as f:
	
	counter = 1
	print('\n', '\n')

	for line in f:
		
		before_keyword, keyword, after_keyword = line.partition(' : ')
		splited = [x.strip() for x in after_keyword.split('/')]
		
		before_keyword2, keyword, after_keyword2 = line.partition(' ____ ')
		previous_word = before_keyword2.split(' ')[-1]
		next_word = after_keyword2.split(' ')[0]

		the_max_uni = 'None'
		the_max_bi = 'None'
		the_max_bi_smo = 'None'

		print("\n", "\n", counter, "Sentence: ", line)

		############### UNIGRAMS: ###############
		initial = 0
		for i in range(0, len(splited)):
			
			if unigram_count[splited[i]] >= initial:
				the_max_uni = splited[i]
				initial = unigram_count[splited[i]]
		print('Using Unigrams: ' , the_max_uni)


		############### BIGRAMS: ###############
		initial = 0
		for i in range(0, len(splited)):
			if (unigram_count[previous_word] != 0 and unigram_count[splited[i]] != 0):
				probability_prev = (bigram_count[(previous_word,splited[i])]) / (unigram_count[previous_word])
				probability_next = (bigram_count[(splited[i]), next_word]) / (unigram_count[splited[i]])
				probability = probability_prev * probability_next

				if (probability != 0 and probability >= initial):
					the_max_bi = splited[i]
					initial = probability
		print('Using Bigrams: ', the_max_bi)

		############### BIGRAMS & SMOOTHING: ###############
		initial = 0
		for i in range(0, len(splited)):
			if (unigram_count[previous_word] != 0 and unigram_count[splited[i]] != 0):
				probability_prev = (bigram_count[(previous_word,splited[i])] + 1) / (unigram_count[previous_word] + len(unigram_count))
				probability_next = (bigram_count[(splited[i]), next_word] + 1) / (unigram_count[splited[i]] + len(unigram_count))
				probability = probability_prev * probability_next

				if (probability !=0 and probability >= initial):
					the_max_bi_smo = splited[i]
					initial = probability
		print('Using Bigrams & Smoothing :', the_max_bi_smo)

		counter += 1





