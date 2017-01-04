#!/usr/bin/env python
import sys
import models
from collections import namedtuple
import optparse
import operator
from collections import defaultdict
optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")

# Two new tuning parameters hav been added m and w
optparser.add_option("-m", "--phrases-per-(i,j) tuple", dest="m", default=1, type="int", help="Limit on number of phrases with higest log probs to consider per french phrase (per i,j value) (default=1)")
optparser.add_option("-w", "--LM look ahead pruning by word prob", dest="w", default=-100, type="float", help="Limit on number of phrases to cnsider)")

opts = optparser.parse_args()[0]
tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
def bitmap(sequence):
  """ Generate a coverage bitmap for a sequence of indexes """
  return reduce(lambda x,y: x|y, map(lambda i: long('1'+'0'*i,2), sequence), 0)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

hypothesis = namedtuple("hypothesis", "j, logprob, lm_state, predecessor, phrase")
phrase_tuple = namedtuple("phrase_tuple","j, phrase, weighted_sum, phrase_prob")

for f in french:
    lm_state = lm.begin()
    lm_look_ahead_table = defaultdict(float)
    for i in range(len(f)):
      for j in xrange(i+1,len(f)+1):
        if f[i:j] in tm:
          for phrase in tm[f[i:j]]:
            sum=0
            for word in phrase.english.split():
              (lm_state, word_logprob) = lm.score(lm_state, word);
              sum += word_logprob 
            lm_look_ahead_table[phrase[0],lm_state]=sum
for f in french:

  phrase_score = [[] for _ in f]
  pruned_phrase_score = [[] for _ in f]
  phrase_list = [{}]
  for i in range(len(f)):
    for j in xrange(i+1,len(f)+1):
      if f[i:j] in tm:
        phrase_logprob=0
        for phrase in tm[f[i:j]]:
          phrase_logprob += phrase.logprob
        phrase_score[i].append((j,phrase,phrase_logprob))

  for i,dummy_val in enumerate(phrase_score):
    for (j,phrase,phrase_logprob) in phrase_score[i]:
      weighted_sum=0
      temp_dict=defaultdict(list)
      (lm_state,word_logprob)=lm.score(lm.begin(), phrase.english.split(' ')[0])
      for word in phrase.english.split(' '):
        (lm_state, word_logprob) = lm.score(lm_state, word)
        weighted_sum += word_logprob
      temp_dict[(j,phrase)]=phrase_tuple(j,phrase,weighted_sum,phrase_logprob)
      for p in sorted(temp_dict.itervalues(),key=lambda p: -p.weighted_sum)[:opts.m]: 
        pruned_phrase_score[i].append((j,phrase,phrase_logprob))
      
  chart = [{} for _ in f] + [{}]
  chart[0][0] = hypothesis(0, 0.0, lm.begin(), None, None)
  for i, b_set in enumerate(chart[:-1]):
    for b in b_set:
      # Phrase candidate pre-sorting, phrases list pruned according to paramter m
      for (j,phrase,phrase_logprob) in pruned_phrase_score[i]:
        curr_h = chart[i][b]
        #LM look-ahead score, if w threshold is exceeded, we discard the expansion without computing the full LM score.
        if (curr_h.logprob + lm_look_ahead_table[phrase,curr_h.lm_state] < opts.w):
          continue
        if bitmap(range(i,j)) & b == 0:
          new_b = bitmap(range(i,j)) | b
          logprob  = curr_h.logprob + phrase_logprob;
          lm_state = curr_h.lm_state
          
          logprob += lm_look_ahead_table[phrase,lm_state]
          logprob += lm.end(lm_state) if j == len(f) else 0.0
          
          new_h = hypothesis(j,logprob, lm_state, curr_h, phrase)
          if new_b not in chart[j] or chart[j][new_b].logprob < new_h.logprob:
            chart[j][new_b] = new_h
        
  winner= chart[len(f)][bitmap(range(len(f)))]
  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)
  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
