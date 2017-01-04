#!/usr/bin/env python
import sys
import models
from collections import namedtuple
import optparse
import argparse
import Queue
from collections import defaultdict
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
hypothesis = namedtuple("hypothesis", "j,logprob, lm_state, predecessor, phrase, swapped")

class Edge:
  def __init__(self, tail, head, score, span, english):
    self.tail = tail
    self.head = head
    self.score = score
    self.span = span
    self.english = english

  def __str__(self):
    return '[%d, %d) --> %s' % (self.span[0], self.span[1], ' '.join(self.english))

  def __repr__(self):
    return str(self)

  def __hash__(self):
    return hash((self.tail, self.head, self.span, self.english))
    
def get_relevant_phrases(f, tm):
	relevant_phrases = set()
	for i in range(len(f)):
		for j in range(i + 1, len(f) + 1):
			french = tuple(f[i:j])
			if french not in tm:
				continue
			for phrase in tm[french]:
				relevant_phrases.add((i, j, tuple(phrase.english.split()), phrase.logprob))
	return relevant_phrases

for f in french:
  State = namedtuple('State', 'lm_state, words_translated, last_contiguous_span, coverage')

def edges_from(state, lenf, phrases, lm, penalties, constraints):
	if state.lm_state == None:
		return

	if state.words_translated == lenf:
		cost = -lm.end(state.lm_state)
		to_state = State(None, lenf, None, state.coverage)
		edge = Edge(state, to_state, cost, None, tuple())
		yield edge
		return

	for i, j, english, phrase_score in phrases:
		if not (j <= state.last_contiguous_span[0] or i >= state.last_contiguous_span[1]):
			continue
		coverage = 0
		for index, k in enumerate(constraints):
			if k >= i and k < j:
				coverage |= 2 ** index

		if (coverage & state.coverage) != 0:
			continue

		words_translated = state.words_translated + j - i
		if words_translated > lenf:
			continue
		cost = -phrase_score
		lm_state = state.lm_state

		for word in english:
			lm_state, word_logprob = lm.score(lm_state, word)
			cost += -word_logprob

		for k in range(i, j):
			cost += penalties[k]

		if i == state.last_contiguous_span[1]:
			last_contiguous_span = (state.last_contiguous_span[0], j)
		elif j == state.last_contiguous_span[0]:
			last_contiguous_span = (i, state.last_contiguous_span[1])
		else:
			last_contiguous_span = (i, j)

		to_state = State(lm_state, words_translated, last_contiguous_span, coverage | state.coverage)
		edge = Edge(state, to_state, cost, (i, j), tuple(english))
		yield edge
def dijkstra(start, goal, edge_generator):
	dist = defaultdict(lambda: float('inf'))
	prev = defaultdict(lambda: None)
	Q = Queue.PriorityQueue()
	Q.put((0, start))
	while not Q.empty():
		k, u = Q.get()
		if k <= dist[u]:
			dist[u] = k
			if u == goal:
				break
			if dist[u] == float('inf'):
				# Goal node is not reachable
				break
			for edge in edge_generator(u):
				v = edge.head
				alt = dist[u] + edge.score
				if alt < dist[v]:
					Q.put((alt, v))
					dist[v] = alt
					prev[v] = edge
	S = []
	u = goal
	while prev[u] != None:
		S.append(prev[u])
		u = prev[u].tail
	S.reverse()
	return dist[goal], S
penalties = [0.0 for _ in f]
constraints = []
alpha = lambda lambdaa: 1.0 / (1.0 + lambdaa)
lambdaa = 0
max_iterations =50
start_state = State(lm.begin(), 0, (0, 0), 0)
prev_score = float('inf')
for iteration in range(max_iterations):
  print "iteration no is ", iteration , "\n"
  relevant_phrases = get_relevant_phrases(f, tm)
  goal_state = State(None, len(f), None, (2 ** len(constraints) - 1))
  edge_generator=lambda state: edges_from(state, len(f), relevant_phrases, lm, penalties, constraints)
  score, path = dijkstra(start_state, goal_state, edge_generator)
  if -score > prev_score:
    lambdaa += 1
    print "Score went up -- decreasing alpha"
  prev_score = -score 

  english = []
  usage = [0 for _ in f]
  for edge in path:
    if edge.span == None:
      continue
    for i in range(*edge.span):
      usage[i] += 1
      english += list(edge.english)
  print '%f\t%s' % (-score, ' '.join(english)), [x for x in list(enumerate(usage)) if x[1] != 0]
  if False not in [usage[i] == 1 for i in range(len(f))]:
    print 'Optimal solution found!'
    break
  else:
    for i in range(len(f)):
      penalties[i] += (usage[i] - 1) * alpha(lambdaa)
  if iteration % 10 == 0:
    ranked_usage = sorted(enumerate(usage), key=lambda (i, c): c, reverse=True)
    possible_constraints = [i for i, c in ranked_usage if i not in constraints and (i + 1) not in constraints and (i - 1) not in constraints and usage[i] > 1]
    if len(possible_constraints) > 0:
      most_overused = possible_constraints[0]
    else:
      possible_constraints = [i for i, c in ranked_usage if i not in constraints and usage[i] > 1]
      assert len(possible_constraints) > 0
      most_overused = possible_constraints[0]
    print "Still not converged... adding constraint on word %d' % most_overused"
    constraints.append(most_overused)

 
