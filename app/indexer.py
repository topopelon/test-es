# Class to index .vec embeddings files (fasttext), apply PCA (sklearn) and store the vectors in elasticsearch

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


class News:
    def __init__(self, title, content, date):
        self.title = title
        self.content = content
        self.date = date


class Indexer:
    def __init__(self, es, index, model_name):
        self.es = es
        self.index = index
        self.model = SentenceTransformer(model_name)

    def create_index(self):
        # Create index
        print("Creating index...")
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
        mappings = {
            "properties": {
                'title': {'type': 'text'},
                "embeddings": {
                    "type": "dense_vector",
                    "dims": self.model.get_sentence_embedding_dimension()
                }
            }
        }
        self.es.indices.create(
            index=self.index, settings=settings, mappings=mappings)
        print("Index created.")

    def index_news(self, news: list[News]):
        # Index embeddings
        print("Indexing news...")
        for news in news:
            doc = {
                "title": news.title,
                "embeddings": self.model.encode(news.title)
            }
            self.es.index(index=self.index, body=doc)
        print("News indexed.")

    def search(self, text):
        # Encode query using model
        query_embedding = self.model.encode(text)
        # Combine title text search with dense vector search using cosine similarity
        query = {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                    "params": {
                        "query_vector": query_embedding.tolist()
                    }
                }
            }
        }

        res = self.es.search(index=self.index, body={
                             "size": 10, "query": query})

        # Interpret results with highest score first
        for hit in res["hits"]["hits"]:
            print(hit["_score"], hit["_source"]["title"])


if __name__ == "__main__":
    es = Elasticsearch(hosts=["http://elasticsearch:9200"], timeout=60)
    # Check if connection is ok
    if es.ping():
        print("Connected to elasticsearch")
    else:
        print("Could not connect to elasticsearch")
    indexer = Indexer(es, "test", "recobo/agriculture-bert-uncased")

    if indexer.es.indices.exists(index="test"):
        indexer.es.indices.delete(index="test")
    indexer.create_index()
    # Create some agriculture news
    news = [
        News("Machines are used in agriculture to plant, cultivate and harvest crops.", "", ""),
        News("Agriculture is the science and art of cultivating plants and livestock.", "", ""),
        News("Nuts are a rich source of energy and nutrients.", "", ""),
        News("The main agricultural products can be broadly grouped into foods, fibers, fuels and raw materials.", "", ""),
        News("Walnuts and almonds are the most popular nuts.", "", ""),
        News("Jute is a long, soft, shiny bast fiber that can be spun into coarse, strong threads.", "", ""),
        News("Corn is a grain first domesticated by indigenous peoples in southern Mexico about 10,000 years ago.", "", ""),
        News("Quinoa and oats are grains that are rich in protein.", "", ""),
    ]
    indexer.index_news(news)
    while True:
        user_input = input("Write a sentence to search: ")
        indexer.search(user_input)
