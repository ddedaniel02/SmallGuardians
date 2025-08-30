import os
import pickle
import faiss
import numpy as np
import pandas as pd


class Embeddings:
  def __init__(self, embedder):
    self.embedder = embedder
    self.load_examples()

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

  def add_corpus(self, dataset_path, task):
      """
      Add examples to the corpus from the dataset_path. The required columns are 'input' and 'classification'. 
      Classification must be boolean variables (1 for Jailbreak and 0 for Benign)
      Args:
          dataset_path (str): Path to the dataset (csv/json/parquet).
          task (str): Examples come from the 'classifier' or 'evaluator' task.
      """

      ext = os.path.splitext(dataset_path)[1].lower()
      if ext == ".csv":
          df = pd.read_csv(dataset_path)
      elif ext == ".json":
          df = pd.read_json(dataset_path)
      elif ext == ".parquet":
          df = pd.read_parquet(dataset_path)
      else:
          raise ValueError("Format not supported. Use csv, json or parquet.")

      if not {"input", "classification"}.issubset(df.columns):
          raise ValueError("The dataset must have 'input' and 'classification' columns.")

      if task == "classifier":
         malicious_index = self.classifier_malicious_index
         benign_index = self.classifier_benign_index
         malicious_examples = self.classifier_malicious_examples
         benign_examples = self.classifier_benign_examples
      elif task == "evaluator":
         malicious_index = self.evaluator_malicious_index
         benign_index = self.evaluator_benign_index
         malicious_examples = self.evaluator_malicious_examples
         benign_examples = self.evaluator_benign_examples
      else:
         raise ValueError("Task must be 'classifier' or 'evaluator'.")
      malicious_inputs = df[df["classification"] == 1]["input"].tolist()
      benign_inputs = df[df["classification"] == 0]["input"].tolist()

      if len(malicious_inputs) > 0:
          malicious_embedder = self.embedder.encode(malicious_inputs, convert_to_numpy=True)
          malicious_index.add(np.array(malicious_embedder, dtype=np.float32))
          malicious_examples.extend(malicious_inputs)
          write_path_malicious_index = f"{task}/malicious/index.faiss"
          faiss.write_index(malicious_index, write_path_malicious_index)
          write_path_malicious_example = f"{task}/malicious/examples.pkl"
          with open(write_path_malicious_example, "wb") as f:
              pickle.dump(malicious_examples, f)

      if len(benign_inputs) > 0:
          benign_embedder = self.embedder.encode(benign_inputs, convert_to_numpy=True)
          benign_index.add(np.array(benign_embedder, dtype=np.float32))
          benign_examples.extend(benign_inputs)
          write_path_benign_index = f"{task}/benign/index.faiss"
          faiss.write_index(benign_index, write_path_benign_index)
          write_path_benign_example = f"{task}/benign/examples.pkl"
          with open(write_path_benign_example, "wb") as f:
              pickle.dump(benign_examples, f)

      self.load_examples()

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

