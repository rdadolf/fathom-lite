#!/usr/bin/env python

import numpy as np

def process_statement(statement, vocab):
  'Turn a question or sentence into a list of words and update vocabulary.'
  # Remove punctuation and capitalization
  bare_statement = statement.translate(None,'?.').lower()
  tokens = bare_statement.strip().split()
  [vocab.add(_) for _ in tokens]
  return tokens, vocab

def load_and_parse(filename):
  vocab = set([])
  triples = [] # A list of (story, question, answer) triples

  with open(filename) as f:
    for line in f.readlines():
      is_question = '\t' in line

      # Split and tokenize
      statement_index,statement = line.strip().split(' ', 1)
      if is_question:
        statement,answer,_ = statement.split('\t')
      statement_tokens,vocab = process_statement(statement, vocab)

      # Collect and store content
      if is_question:
        triples.append( (story,statement_tokens,answer) )
      else:
        if statement_index=='1':
          story = [statement_tokens]
        else:
          story.append(statement_tokens)
  return triples, vocab

def encode(triples, dictionary):
  'Replace every word in a Q/A triple with a number.'
  encoded_triples = []
  for (story,question,answer) in triples:
    encode = lambda w: dictionary[w]
    encoded_triples.append(
      ([map(encode,sentence) for sentence in story],
       map(encode,question),
       encode(answer)))
  return encoded_triples

def arrays_from_encoded_triples(encoded_triples):
  n_triples = len(encoded_triples)
  sentence_size = max([max([len(story) for story in stories]) for stories,_,_ in encoded_triples])
  question_size = max([len(question) for _,question,_ in encoded_triples])
  story_size = max([len(stories) for stories,_,_ in encoded_triples])

  stories = np.zeros( (n_triples, story_size, sentence_size), dtype=int )
  questions = np.zeros( (n_triples, sentence_size), dtype=int )
  answers = np.zeros( (n_triples), dtype=int )

  for i,(story,question,answer) in enumerate(encoded_triples):
    for j,sentence in enumerate(story):
      stories[i,j,0:len(sentence)] = sentence
    questions[i,0:len(question)] = question
    answers[i] = answer

  return stories,questions,answers
  

def load_to_npz(source_filename, dest_filename_prefix):
  triples,vocab = load_and_parse(source_filename)
  dictionary = { word:index+1 for index,word in enumerate(vocab) }
  dictionary[None] = 0 # Add NIL word
  encoded_triples = encode(triples, dictionary)
  stories,questions,answers = arrays_from_encoded_triples(encoded_triples)
  np.savez(dest_filename_prefix+'-stories.npz', stories)
  np.savez(dest_filename_prefix+'-questions.npz', questions)
  np.savez(dest_filename_prefix+'-answers.npz', answers)

if __name__=='__main__':
  data_path = 'tasks_1-20_v1-2/en/'
  load_to_npz(data_path+'qa1_single-supporting-fact_train.txt',
              'babi')
