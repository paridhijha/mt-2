#!/usr/bin/env python
import optparse
import sys
import models
import math

# Three little utility functions:
def bitmap(sequence):
  """ Generate a coverage bitmap for a sequence of indexes """
  return reduce(lambda x,y: x|y, map(lambda i: long('1'+'0'*i,2), sequence), 0)

def bitmap2str(b, n, on='o', off='.'):
  """ Generate a length-n string representation of bitmap b """
  return '' if n==0 else (on if b&1==1 else off) + bitmap2str(b>>1, n-1, on, off)

def logadd10(x,y):
  """ Addition in logspace (base 10): if x=log(a) and y=log(b), returns log(a+b) """
  return x + math.log10(1 + pow(10,y-x))

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-v", "--verbosity", dest="verbosity", default=1, type="int", help="Verbosity level, 0-3 (default=1)")
opts = optparser.parse_args()[0]
tm = models.TM(opts.tm,sys.maxint)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()]
english = [tuple(line.strip().split()) for line in sys.stdin]
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

def maybe_write(s, verbosity):
  if opts.verbosity > verbosity:
    sys.stdout.write(s)
    sys.stdout.flush()

total_logprob = 0.0
unaligned_sentences = 0
for sent_num, (f, e) in enumerate(zip(french, english)):
  lm_state = lm.begin()
  lm_logprob = 0.0
  for word in e + ("</s>",):
    total_logprob += lm_logprob
  
  alignments = [[] for _ in e]
  for fi in xrange(len(f)):
    for fj in xrange(fi+1,len(f)+1):
      if f[fi:fj] in tm:
        for phrase in tm[f[fi:fj]]:
          ephrase = tuple(phrase.english.split())
          for ei in xrange(len(e)+1-len(ephrase)):
            ej = ei+len(ephrase)
            if ephrase == e[ei:ej]:
              alignments[ei].append((ej, phrase.logprob, fi, fj))

  # Compute sum of probability of all possible alignments by dynamic programming.
  # To do this, recursively compute the sum over all possible alignments for each
  # each pair of English prefix (indexed by ei) and French coverage (indexed by 
  # bitmap v), working upwards from the base case (ei=0, v=0) [i.e. forward chaining]. 
  # The final sum is the one obtained for the pair (ei=len(e), v=range(len(f))
  chart = [{} for _ in e] + [{}]
  chart[0][0] = 0.0
  for ei, sums in enumerate(chart[:-1]):
    for v in sums:
      for ej, logprob, fi, fj in alignments[ei]:
        if bitmap(range(fi,fj)) & v == 0:
          new_v = bitmap(range(fi,fj)) | v
          if new_v in chart[ej]:
            chart[ej][new_v] = logadd10(chart[ej][new_v], sums[v]+logprob)
          else:
            chart[ej][new_v] = sums[v]+logprob
  goal = bitmap(range(len(f)))
  if goal in chart[len(e)]:
    total_logprob += chart[len(e)][goal]
  else:
    unaligned_sentences += 1

sys.stdout.write("\nTotal corpus log probability (LM+TM): %f\n" % total_logprob)
if (len(french) != len(english)):
  sys.stdout.write("ERROR: French and English files are not the same length! Only complete output can be graded!\n")
if unaligned_sentences > 0:
  sys.stdout.write("ERROR: There were %d unaligned sentences! Only sentences that align under the model can be graded!\n" % unaligned_sentences)
if (len(french) != len(english)) or unaligned_sentences > 0:
  sys.exit(1) # signal problem to caller


#  maybe_write("===========================================================\n",1)
#  maybe_write("SENTENCE PAIR:\n%s\n%s\n" % (" ".join(f), " ".join(e)),0)

#  maybe_write("\nLANGUAGE MODEL SCORES:\n",1)
#maybe_write("Aligning...\n",0)
#maybe_write("NOTE: TM logprobs may be positive since they do not include segmentation\n",0)
#    maybe_write("%s: " % " ".join(lm_state + (word,)),1)
#    (lm_state, word_logprob) = lm.score(lm_state, word)
#    lm_logprob += word_logprob
#    maybe_write("%f\n" % (word_logprob,),1)
#  maybe_write("TOTAL LM LOGPROB: %f\n" % lm_logprob,0)
