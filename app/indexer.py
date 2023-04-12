# Class to index .vec embeddings files (fasttext), apply PCA (sklearn) and store the vectors in elasticsearch

from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from elasticsearch import Elasticsearch
import time


class News:
    def __init__(self, title, content, date):
        self.title = title
        self.content = content
        self.date = date


class Indexer:
    def __init__(self, es: Elasticsearch, index, embedding_file, pca_n_components=50, pca_whiten=True):
        self.es = es
        self.index = index
        self.embedding_file = embedding_file
        self.pca_n_components = pca_n_components
        self.pca_whiten = pca_whiten
        self.model = None

    def load_model_to_memory(self, apply_pca=False):
        # Load trained model in memory
        print("Loading model...")
        self.model = KeyedVectors.load_word2vec_format(
            self.embedding_file, binary=False)
        print("model loaded.")

        # Apply PCA
        if apply_pca:
            print("Applying PCA...")
            pca = PCA(n_components=self.pca_n_components,
                      whiten=self.pca_whiten)
            self.model.vectors = pca.fit_transform(self.model.vectors)
            print("PCA applied.")

    def create_index(self):
        # Create index
        print("Creating index...")

        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
        mappings = {
            "properties": {
                "title": {
                    "type": "text"
                },
                "embeddings": {
                    "type": "dense_vector",
                    "dims": self.model.vectors.shape[1]
                }
            }
        }
        self.es.indices.create(
            index=self.index, settings=settings, mappings=mappings)

        print("Index created.")

    def index_embeddings(self, news: list[News]):
        # Index embeddings
        print("Indexing embeddings...")
        for news in news:
            # Average embeddings
            words = []
            title_lowercase = news.title.lower()
            for word in title_lowercase.split():
                if word in self.model:
                    words.append(word)
            embeddings = self.model[words].mean(
                axis=0).tolist()
            doc = {
                "title": news.title,
                "embeddings": embeddings
            }
            self.es.index(index=self.index, body=doc)

        print("Embeddings indexed.")

    def search(self, title):
        # Search
        print("Searching...")
        # Make query with embeddings
        query = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                        "params": {
                            "query_vector": self.model[title.lower().split()].mean(
                                axis=0).tolist()
                        }
                    }
                }
            }
        }
        res = self.es.search(index=self.index, body=query)
        print("Search done.")
        return res


if __name__ == "__main__":
    es = Elasticsearch(hosts=["http://elasticsearch:9200"])
    # Check if connection is ok
    if es.ping():
        print("Connected to elasticsearch")
    else:
        print("Could not connect to elasticsearch")
    indexer = Indexer(es, "test", "data/wiki.es.vec")

    while True:
        user_input = input("Load model to memory?:(y/n) ")
        if user_input == "y":
            user_input = input("Apply PCA algorithm?:(y/n) ")
            indexer.load_model_to_memory(apply_pca=user_input == "y")
        if indexer.es.indices.exists(index="test"):
            indexer.es.indices.delete(index="test")
        indexer.create_index()
        news = [News("Fribin tira con fuerza de la locomotora agroindustrial", "", ""), News("Riego inteligente para exportar mejores cerezas", "", ""), News("El dulce más popular y viajero de Barbastro celebra su 120 aniversario", "", ""), News(
            "Nuevos tiempos para el conejo en Hispania", "", ""), News("Aniñón sabe de cerezas", "", ""), News("Viñas de la Denominación de Origen Campo de Borja", "", "")]
        indexer.index_embeddings(news)
        time.sleep(5)
        user_input = input("Write a sentence to search: ")
        res = indexer.search(user_input)
        # Interpret results with highes score first
        for hit in res["hits"]["hits"]:
            print(hit["_score"], hit["_source"]["title"])
