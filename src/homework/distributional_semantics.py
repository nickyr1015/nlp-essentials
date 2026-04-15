import numpy as np

#helper function to read the words in doc
def read_word_embeddings(path):
    embeddings = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            word = parts[0]

            try:
                vector = np.array([float(x) for x in parts[1:]], dtype=float)
            except ValueError:
                raise ValueError(f"Invalid float values on line {line_num}: {line}")

            embeddings[word] = vector

    return embeddings

#cosine sim help function 
def cosine_similarity(vec1, vec2):

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def similar_words(word_embeddings, target_word, threshold):

    if target_word not in word_embeddings:
        return []

    target_vec = word_embeddings[target_word]
    results = []

    for word, vec in word_embeddings.items():
        if word == target_word:
            continue

        sim = cosine_similarity(target_vec, vec)
        if sim >= threshold:
            results.append((word, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def document_similarity(word_embeddings, doc1, doc2):
   
    def get_document_embedding(doc):
        words = doc.split()
        vectors = [word_embeddings[word] for word in words if word in word_embeddings]

        if not vectors:
            # If no words from the document are in the embeddings, return zero vector
            dim = len(next(iter(word_embeddings.values())))
            return np.zeros(dim, dtype=float)

        return np.mean(vectors, axis=0)

    doc1_vec = get_document_embedding(doc1)
    doc2_vec = get_document_embedding(doc2)

    return cosine_similarity(doc1_vec, doc2_vec)

if __name__ == "__main__":


    embeddings = read_word_embeddings("dat/word_embeddings.txt")

    print("test 1")
    print(type(embeddings))
    print(len(embeddings))
    print("computer" in embeddings)
    print(type(embeddings["computer"]))
    print(embeddings["computer"].shape)

    print("test 2")
    results = similar_words(embeddings, "computer", 0.5)
    print(results[:10])  

    results = similar_words(embeddings, "computer", 0.5)
    for i in range(len(results) - 1):
        assert results[i][1] >= results[i + 1][1]
    print("sorted correctly")   


    print("test 3")
    print(document_similarity(embeddings, "computer software hardware", "computer software"))
    print(document_similarity(embeddings, "coffee tea sugar", "computer software website"))
    print(document_similarity(embeddings, "Atlanta Seattle Chicago", "Houston Dallas Miami"))