import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import networkx as nx
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


class Document:
    def __init__(self, relevance: int, qid: int, features: Dict[int, float]):
        self.relevance = relevance
        self.qid = qid
        self.features = features

    def get_feature(self, feature_id: int) -> float:
        return self.features.get(feature_id, 0.0)


class DataLoader:

    @staticmethod
    def parse_line(line: str) -> Document:
        parts = line.strip().split()
        relevance = int(parts[0])
        qid = int(parts[1].split(":")[1])

        features = {}
        for part in parts[2:]:
            if ":" in part:
                feature_id, value = part.split(":")
                features[int(feature_id)] = float(value)

        return Document(relevance, qid, features)

    @staticmethod
    def load_data(filename: str) -> List[Document]:
        documents = []
        with open(filename, "r") as f:
            for line in f:
                if line.strip():
                    documents.append(DataLoader.parse_line(line))
        return documents

    @staticmethod
    def get_documents_by_qid(documents: List[Document], qid: int) -> List[Document]:
        return [doc for doc in documents if doc.qid == qid]


class DCGEvaluator:
    @staticmethod
    def dcg_at_k(relevances: List[int], k: int = -1) -> float:
        if k == -1:
            k = len(relevances)

        dcg = 0.0
        for i, rel in enumerate(relevances[:k]):
            dcg += (2**rel - 1) / math.log2(i + 2)
        return dcg

    @staticmethod
    def ndcg_at_k(relevances: List[int], k: int = -1) -> float:
        if k == -1:
            k = len(relevances)

        dcg = DCGEvaluator.dcg_at_k(relevances, k)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = DCGEvaluator.dcg_at_k(ideal_relevances, k)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def create_ideal_ranking(documents: List[Document]) -> List[Document]:
        return sorted(documents, key=lambda doc: doc.relevance, reverse=True)


class FeatureRanker:
    def __init__(self, feature_id: int):
        self.feature_id = feature_id

    def rank_documents(self, documents: List[Document]) -> List[Document]:
        return sorted(
            documents, key=lambda doc: doc.get_feature(self.feature_id), reverse=True
        )


class BM25Ranker:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def rank_documents(self, documents: List[Document]) -> List[Document]:
        scored_docs = []

        for doc in documents:
            tf = doc.get_feature(1)
            doc_len = max(doc.get_feature(11), 1)
            score = (tf * self.k1) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / 1000)
            )
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]


class EvaluationMetrics:
    @staticmethod
    def precision_at_k(relevances: List[int], k: int) -> float:
        if k == 0:
            return 0.0
        relevant_count = sum(1 for rel in relevances[:k] if rel > 0)
        return relevant_count / k

    @staticmethod
    def recall_at_k(relevances: List[int], k: int, total_relevant: int) -> float:
        if total_relevant == 0:
            return 0.0
        relevant_count = sum(1 for rel in relevances[:k] if rel > 0)
        return relevant_count / total_relevant

    @staticmethod
    def average_precision(relevances: List[int]) -> float:
        if not relevances:
            return 0.0

        total_relevant = sum(1 for rel in relevances if rel > 0)
        if total_relevant == 0:
            return 0.0

        ap = 0.0
        relevant_count = 0

        for i, rel in enumerate(relevances):
            if rel > 0:
                relevant_count += 1
                ap += relevant_count / (i + 1)

        return ap / total_relevant


class LinkAnalyzer:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph_from_file(self, filename: str):
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    source, target = map(int, line.split())
                    self.graph.add_edge(source, target)

    def compute_pagerank(
        self, alpha: float = 0.85, max_iter: int = 100
    ) -> Dict[int, float]:
        return nx.pagerank(self.graph, alpha=alpha, max_iter=max_iter)

    def compute_hits(
        self, max_iter: int = 100
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        hubs, authorities = nx.hits(self.graph, max_iter=max_iter)
        return hubs, authorities

    def get_in_degree_distribution(self) -> Dict[int, int]:
        in_degrees = dict(self.graph.in_degree())
        degree_counts = Counter(int(deg) for deg in in_degrees.values())
        return dict(degree_counts)


class ProbabilisticRetrieval:
    def __init__(self, mu: float = 1000):
        self.mu = mu
        self.collection_stats = {}

    def build_collection_statistics(self, documents: List[Document]):
        total_terms = 0
        term_counts = defaultdict(int)

        for doc in documents:
            tf = doc.get_feature(1)
            total_terms += tf
            term_counts[1] = term_counts.get(1, 0) + tf

        self.collection_stats = {
            "total_terms": total_terms,
            "term_counts": dict(term_counts),
        }

    def score_document(self, doc: Document) -> float:
        doc_tf = doc.get_feature(1)
        doc_len = max(doc.get_feature(11), 1)

        coll_tf = self.collection_stats.get("term_counts", {}).get(1, 0)
        coll_len = self.collection_stats.get("total_terms", 1)

        numerator = doc_tf + self.mu * (coll_tf / coll_len)
        denominator = doc_len + self.mu

        if denominator == 0:
            return 0.0

        return math.log(numerator / denominator)

    def rank_documents(self, documents: List[Document]) -> List[Document]:
        scored_docs = [(doc, self.score_document(doc)) for doc in documents]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]


def main():
    documents = DataLoader.load_data("data.txt")

    target_qid = 46
    print(f"Using qid:{target_qid}")

    qid_documents = DataLoader.get_documents_by_qid(documents, target_qid)
    print(f"Found {len(qid_documents)} documents for qid:{target_qid}")

    print("\n=== Question 1: Ideal DCG Ranking ===")

    ideal_ranking = DCGEvaluator.create_ideal_ranking(qid_documents)

    print("Top 5 ideal ranking entries:")
    for doc in ideal_ranking[:5]:
        features_str = " ".join([f"{k}:{v}" for k, v in sorted(doc.features.items())])
        print(f"{doc.relevance} qid:{doc.qid} {features_str}")

    relevances = [doc.relevance for doc in ideal_ranking]
    ndcg_50 = DCGEvaluator.ndcg_at_k(relevances, 50)
    ndcg_full = DCGEvaluator.ndcg_at_k(relevances)

    print(f"nDCG@50: {ndcg_50:.4f}")
    print(f"nDCG (full list): {ndcg_full:.4f}")

    print("\n=== Question 2: Feature-75 Ranker ===")

    feature_ranker = FeatureRanker(75)
    feature_ranking = feature_ranker.rank_documents(qid_documents)
    feature_relevances = [doc.relevance for doc in feature_ranking]

    total_relevant = sum(1 for rel in feature_relevances if rel > 0)
    p_at_10 = EvaluationMetrics.precision_at_k(feature_relevances, 10)
    r_at_10 = EvaluationMetrics.recall_at_k(feature_relevances, 10, total_relevant)
    avg_precision = EvaluationMetrics.average_precision(feature_relevances)
    ndcg_at_20 = DCGEvaluator.ndcg_at_k(feature_relevances, 20)

    print(f"P@10: {p_at_10:.4f}")
    print(f"R@10: {r_at_10:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"nDCG@20: {ndcg_at_20:.4f}")

    print("\n=== Question 3: BM25 Baseline ===")

    bm25_ranker = BM25Ranker()
    bm25_ranking = bm25_ranker.rank_documents(qid_documents)
    bm25_relevances = [doc.relevance for doc in bm25_ranking]

    bm25_p_at_10 = EvaluationMetrics.precision_at_k(bm25_relevances, 10)
    bm25_r_at_10 = EvaluationMetrics.recall_at_k(bm25_relevances, 10, total_relevant)
    bm25_avg_precision = EvaluationMetrics.average_precision(bm25_relevances)

    print(f"BM25 P@10: {bm25_p_at_10:.4f}")
    print(f"BM25 R@10: {bm25_r_at_10:.4f}")
    print(f"BM25 Average Precision: {bm25_avg_precision:.4f}")

    print("\n=== Question 4: Link Analysis ===")

    try:
        link_analyzer = LinkAnalyzer()
        link_analyzer.build_graph_from_file("web-Google.txt")

        pagerank_scores = link_analyzer.compute_pagerank()
        hubs, authorities = link_analyzer.compute_hits()
        in_degree_dist = link_analyzer.get_in_degree_distribution()
        degrees = sorted(in_degree_dist.keys())
        counts = [in_degree_dist[d] for d in degrees]

        plt.figure(figsize=(10, 6))
        plt.loglog(degrees, counts, "bo-", alpha=0.7)
        plt.xlabel("In-degree")
        plt.ylabel("Number of nodes")
        plt.title("In-degree Distribution (log-log scale)")
        plt.grid(True, alpha=0.3)
        plt.savefig("deg_dist.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("In-degree distribution plot saved as deg_dist.png")

        nodes_with_both = []
        for node in link_analyzer.graph.nodes():
            if node in pagerank_scores:
                in_deg = link_analyzer.graph.in_degree(node)
                nodes_with_both.append((pagerank_scores[node], in_deg))

        if nodes_with_both:
            pagerank_vals, in_deg_vals = zip(*nodes_with_both)
            correlation, p_value = spearmanr(pagerank_vals, in_deg_vals)
            print(
                f"Spearman correlation between PageRank and in-degree: {correlation:.4f}"
            )

    except FileNotFoundError:
        print("web-Google.txt not found. Skipping link analysis.")

    print("\n=== Question 5: Probabilistic Retrieval ===")

    prob_retrieval = ProbabilisticRetrieval()
    prob_retrieval.build_collection_statistics(documents)
    prob_ranking = prob_retrieval.rank_documents(qid_documents)
    prob_relevances = [doc.relevance for doc in prob_ranking]

    prob_p_at_10 = EvaluationMetrics.precision_at_k(prob_relevances, 10)
    prob_r_at_10 = EvaluationMetrics.recall_at_k(prob_relevances, 10, total_relevant)
    prob_avg_precision = EvaluationMetrics.average_precision(prob_relevances)

    print(f"Probabilistic P@10: {prob_p_at_10:.4f}")
    print(f"Probabilistic R@10: {prob_r_at_10:.4f}")
    print(f"Probabilistic Average Precision: {prob_avg_precision:.4f}")

    print("\n=== Summary Table ===")
    print("Method\t\tP@10\t\tR@10\t\tAvgP")
    print("-" * 50)
    print(f"Feature-75\t{p_at_10:.4f}\t\t{r_at_10:.4f}\t\t{avg_precision:.4f}")
    print(
        f"BM25\t\t{bm25_p_at_10:.4f}\t\t{bm25_r_at_10:.4f}\t\t{bm25_avg_precision:.4f}"
    )
    print(
        f"Probabilistic\t{prob_p_at_10:.4f}\t\t{prob_r_at_10:.4f}\t\t{prob_avg_precision:.4f}"
    )


if __name__ == "__main__":
    main()
