import os
import pickle
import faiss
import numpy as np
from datasets import load_dataset


class Embeddings:
  def __init__(self, embedder):
    self.embedder = embedder

  def reading_files(self, path):
    index_path = os.path.join(path, "index.faiss")
    examples_path = os.path.join(path, "examples.pkl")
    index = faiss.read_index(index_path)
    with open(examples_path, "rb") as f:
        examples = pickle.load(f)
    return index, examples


  def load_examples(self):
    # Loading classifier attributes from disk
    self.classifier_malicious_index, self.classifier_malicious_examples = self.reading_files("classifier/malicious")
    self.classifier_benign_index, self.classifier_benign_examples = self.reading_files("classifier/benign")

    self.evaluator_malicious_index, self.evaluator_malicious_examples = self.reading_files("evaluator/malicious")
    self.evaluator_benign_index, self.evaluator_benign_examples = self.reading_files("evaluator/benign")

  # TO DO: CHANGE AND FIX THIS FUNCTION
  # def add_corpus(self,task, dataset_name, split, prompt_column_name, label_column_name):
  #   malicious_examples = []
  #   benign_examples = []
  #   dataset = load_dataset(dataset_name)
  #   for row in dataset[split]:
  #     new_corpus.append(row[prompt_column_name])
  #     new_labels.append(row[label_column_name])

  #   classifier_embeddings = self.embedder.encode(new_corpus, convert_to_numpy=True)
  #   if task == "classifier":
  #     self.classifier_examples.extend(new_corpus)
  #     self.classifier_labels.extend(new_labels)
  #     self.classifier_index.add(classifier_embeddings)

  #     faiss.write_index(self.classifier_index, INDEX_CLASSIFIER_PATH)
  #     with open(LABELS_CLASSIFIER_PATH, "wb") as f:
  #         pickle.dump(self.classifier_labels, f)
  #     with open(EXAMPLES_CLASSIFIER_PATH, "wb") as f:
  #         pickle.dump(self.classifier_examples, f)
  #   elif task == "evaluator":
  #     self.evaluator_examples.extend(new_corpus)
  #     self.evaluator_labels.extend(new_labels)
  #     self.evaluator_index.add(classifier_embeddings)

  #     faiss.write_index(self.evaluator_index, INDEX_EVALUATOR_PATH)
  #     with open(LABELS_EVALUATOR_PATH, "wb") as f:
  #         pickle.dump(self.evaluator_labels, f)
  #     with open(EXAMPLES_EVALUATOR_PATH, "wb") as f:
  #         pickle.dump(self.evaluator_examples, f)
  # TO FINISH
  @staticmethod
  def truncate(text, max_len=500):
      if text is None:
          return ""
      s = str(text)
      if max_len <= 0:
          return "…" if s else ""
      if len(s) <= max_len:
          return s
      cut = s.rfind(" ", 0, max_len)
      if cut != -1 and cut >= int(max_len * 0.6):
          s = s[:cut]
      else:
          s = s[:max_len]
      return s.rstrip() + "…"



  def retrieve_similar(self, task, prompt, k=1, return_distances=False):
      query_vec = self.embedder.encode([prompt], convert_to_numpy=True)
      if task == "classifier":
        malicious_distances, malicious_index = self.classifier_malicious_index.search(query_vec, k)
        benign_distances, benign_index = self.classifier_benign_index.search(query_vec, k)
        if return_distances:
           return (malicious_distances, benign_distances)
        return (
            [self.truncate(self.classifier_malicious_examples[int(i)]) for i in malicious_index[0] if i != -1],
            [self.truncate(self.classifier_benign_examples[int(i)])    for i in benign_index[0]    if i != -1],
        )

      elif task == "evaluator":
        malicious_distances, malicious_index = self.evaluator_malicious_index.search(query_vec, k)
        benign_distances, benign_index = self.evaluator_benign_index.search(query_vec, k)
        if return_distances:
          return (malicious_distances, benign_distances)
        return (
            [self.truncate(self.evaluator_malicious_examples[int(i)]) for i in malicious_index[0] if i != -1],
            [self.truncate(self.evaluator_benign_examples[int(i)])    for i in benign_index[0]    if i != -1],
        )

