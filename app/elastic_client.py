from elasticsearch import Elasticsearch
import logging
import streamlit as st
from config import INDEX_NAME, ES_HOSTS, ES_TIMEOUT

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

@st.cache_resource
def get_elasticsearch_client():
    """Crée et met en cache le client Elasticsearch."""
    try:
        es_client = Elasticsearch(ES_HOSTS, request_timeout=ES_TIMEOUT)
        if not es_client.ping():
            st.error("❌ Connexion à Elasticsearch échouée.")
            return None
        logging.info("Connecté à Elasticsearch avec succès.")
        return es_client
    except Exception as e:
        st.error(f"❌ Erreur de connexion à Elasticsearch. Vérifiez le serveur : {e}")
        return None

def search_similar_images(es_client, query_features, metric, top_n=5, tag_query=None):
    """
    Effectue la recherche vectorielle, textuelle, ou les deux (le texte agit comme filtre).
    NOTE: La logique de filtrage stricte est conservée ici.
    """
    if not es_client:
        return []
    
    tag_match_clause = None
    if tag_query:
        tag_match_clause = {
            "bool": {
                "should": [
                    {"match": {"tags": {"query": tag_query, "boost": 2}}},
                    {"fuzzy": {"tags": {"value": tag_query, "fuzziness": "AUTO"}}}
                ],
                "minimum_should_match": 1
            }
        }
    
    if query_features.size > 0:
        query_for_script_score = tag_match_clause if tag_match_clause else {"match_all": {}}

        final_query_body = {
            "script_score": {
                "query": query_for_script_score,
                "script": {
                    "source": metric,
                    "params": {"query_vector": query_features.tolist()}
                }
            }
        }
    
    elif tag_query:
        final_query_body = tag_match_clause 
        
    else:
        return []

    query = {
        "size": top_n,
        "_source": ["image_id", "image_name", "relative_path", "tags"], 
        "query": final_query_body
    }
    
    try:
        response = es_client.search(index=INDEX_NAME, body=query)
        return [
            {
                'score': hit['_score'],
                'image_id': hit['_source']['image_id'],
                'image_name': hit['_source']['image_name'],
                'relative_path': hit['_source']['relative_path'],
                'tags': hit['_source'].get('tags', []) 
            }
            for hit in response['hits']['hits']
        ]
    except Exception as e:
        st.error(f"Erreur lors de la recherche Elasticsearch : {e}")
        return []