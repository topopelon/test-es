# test-es

## IMPORTANT

Add wiki.es.vec under app/data to run the app and name it corpus.vec

Download from: <https://github.com/dccuchile/spanish-word-embeddings>


## Running the indexer

Run:

```
docker compose up --build
```

Once it is running on docker, enter the python app container and execute:

```
python3 indexer.py
```

If you cannot attach to the container in vscode (https://code.visualstudio.com/docs/devcontainers/containers), then execute:

```
docker exec -it <container name or id> python3 indexer.py
```
