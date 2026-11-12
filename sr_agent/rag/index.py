"""Index RAG hybride sur les textes intégraux.

- découpage en passages avec chevauchement (métadonnées : article, position) ;
- recherche dense (FAISS, embeddings sentence-transformers) + lexicale (BM25) ;
- fusion par Reciprocal Rank Fusion, puis re-ranking optionnel (cross-encoder) ;
- compression du contexte : on ne garde que les phrases du passage les plus proches de la requête.
Index sauvegardé sur disque -> pas de recalcul après une coupure.
"""
import json, re, pathlib
import numpy as np

EMB_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def chunk_text(text, size=1200, overlap=200):
    text = re.sub(r"\n{3,}", "\n\n", text)
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):                      # couper sur une fin de phrase si possible
            k = text.rfind(". ", i + size // 2, j)
            j = k + 1 if k > 0 else j
        out.append(text[i:j].strip())
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return [c for c in out if len(c) > 100]


def _tok(t):
    return re.findall(r"[a-z0-9]+", t.lower())


class HybridIndex:
    def __init__(self, folder):
        self.dir = pathlib.Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunks, self._emb, self._faiss, self._bm25, self._rerank = [], None, None, None, None

    # ---------- modèles (chargés à la demande) ----------
    def _embedder(self):
        if self._emb is None:
            from sentence_transformers import SentenceTransformer
            self._emb = SentenceTransformer(EMB_MODEL)
        return self._emb

    def _encode(self, texts):
        return np.asarray(self._embedder().encode(texts, batch_size=64, normalize_embeddings=True,
                                                  show_progress_bar=len(texts) > 200), dtype="float32")

    # ---------- construction / chargement ----------
    def build(self, docs: dict[str, str]):
        import faiss
        self.chunks = [{"paper_id": pid, "pos": i, "text": c}
                       for pid, txt in docs.items() for i, c in enumerate(chunk_text(txt))]
        vecs = self._encode([c["text"] for c in self.chunks])
        self._faiss = faiss.IndexFlatIP(vecs.shape[1])
        self._faiss.add(vecs)
        faiss.write_index(self._faiss, str(self.dir / "dense.faiss"))
        (self.dir / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        self._init_bm25()
        return len(self.chunks)

    def load(self):
        import faiss
        self.chunks = json.loads((self.dir / "chunks.json").read_text(encoding="utf-8"))
        self._faiss = faiss.read_index(str(self.dir / "dense.faiss"))
        self._init_bm25()
        return self

    def exists(self):
        return (self.dir / "dense.faiss").exists() and (self.dir / "chunks.json").exists()

    def _init_bm25(self):
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([_tok(c["text"]) for c in self.chunks])

    # ---------- recherche ----------
    def search(self, query, k=6, paper_id=None, rerank=True, compress=True, pool=40):
        allowed = None if paper_id is None else {i for i, c in enumerate(self.chunks) if c["paper_id"] == paper_id}
        if allowed is not None and not allowed:
            return []
        qv = self._encode([query])
        n_dense = len(self.chunks) if allowed is not None else min(pool * 3, len(self.chunks))
        _, I = self._faiss.search(qv, n_dense)
        dense = [i for i in I[0] if i >= 0 and (allowed is None or i in allowed)][:pool]
        bm = self._bm25.get_scores(_tok(query))
        cand = allowed if allowed is not None else range(len(self.chunks))
        sparse = sorted(cand, key=lambda i: -bm[i])[:pool]
        # Reciprocal Rank Fusion
        rrf = {}
        for ranking in (dense, sparse):
            for r, i in enumerate(ranking):
                rrf[i] = rrf.get(i, 0) + 1 / (60 + r)
        top = sorted(rrf, key=lambda i: -rrf[i])[:max(k * 3, k)]
        if rerank and len(top) > k:
            top = self._rerank_ids(query, top)
        hits = [{**self.chunks[i], "score": float(rrf[i])} for i in top[:k]]
        if compress:
            for h in hits:
                h["text"] = self._compress(query, h["text"])
        return hits

    def _rerank_ids(self, query, ids):
        try:
            if self._rerank is None:
                from sentence_transformers import CrossEncoder
                self._rerank = CrossEncoder(RERANK_MODEL)
            scores = self._rerank.predict([(query, self.chunks[i]["text"]) for i in ids])
            return [i for _, i in sorted(zip(scores, ids), key=lambda x: -x[0])]
        except Exception:
            return ids                          # re-ranking indisponible : on garde l'ordre RRF

    def _compress(self, query, text, keep=4):
        sents = re.split(r"(?<=[.!?])\s+", text)
        if len(sents) <= keep:
            return text
        sv = self._encode(sents)
        qv = self._encode([query])[0]
        best = sorted(np.argsort(-(sv @ qv))[:keep])
        return " … ".join(sents[i] for i in best)
