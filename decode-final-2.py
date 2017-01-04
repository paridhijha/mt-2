#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]
tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]
for f in french:
  # The hypothesis has extra parameter-swapped which stores whether this hypothesis was swapped before or not
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, swapped")
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None,'false')
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  for i, stack in enumerate(stacks[:-1]):
    for h1 in sorted(stack.itervalues(),key=lambda h1: -h1.logprob)[:opts.s]: 
      for j in xrange(i+1,len(f)+1):
        if f[i:j] in tm:
          for phrase in tm[f[i:j]]:
            logprob = h1.logprob + phrase.logprob
            lm_state = h1.lm_state
            for word in phrase.english.split():
              (lm_state, word_logprob) = lm.score(lm_state, word)
              logprob += word_logprob
            logprob += lm.end(lm_state) if j == len(f) else 0.0
            # The topmost hypothesis is h0, which is added on top of h1 which in turn was added before on top of h2
            h0 = hypothesis(logprob, lm_state, h1, phrase,'false')
            h2 = h1.predecessor
            if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob:
              stacks[j][lm_state] = h0
            # check the swap the adjacent phrases only and proceed with swap of h0 and h1 only if h1 was not swapped with h2 before.
            if h1.swapped == 'false':
              if h2 == None: # swapping the first 2 phrases h1 and h0 of f
                logprob=phrase.logprob
                lm_state = lm.begin()
              else:
                logprob=h2.logprob+phrase.logprob
                lm_state = h2.lm_state
              for word in phrase.english.split():
                (lm_state, word_logprob) = lm.score(lm_state, word)
                logprob += word_logprob
              # alt_h1 and alt_h0 will be the alternate hypothesis once h1 and h0 are swapped
              alt_h1 = hypothesis(logprob, lm_state, h2, phrase,'true')
              lm_state=alt_h1.lm_state
              if h2 != None: # if h2 is None and h1 is the initial_hypothesis, donot iterate phrase for word probs
                for word in h1.phrase.english.split():
                  (lm_state, word_logprob) = lm.score(lm_state, word)
                  logprob += word_logprob
              logprob += lm.end(lm_state) if j == len(f) else 0.0
              alt_h0 = hypothesis(logprob,lm_state,alt_h1,h1.phrase,'true')
              if  lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob:
                # alt_h0 is placed at same j , since sum of lengths of both phrases is still same after swapping
                stacks[j][alt_h0.lm_state] = alt_h0
  for a in stacks[-1].itervalues():
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)
  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

