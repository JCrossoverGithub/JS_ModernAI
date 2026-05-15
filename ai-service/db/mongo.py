"""
db/mongo.py — MongoDB client singleton and collection helpers.

Chapter 3 (Data Engineering): provides the data warehouse connection.
All collections live in one database (settings.mongodb_database).

Collections:
  raw_documents   ← RawDocument records written at upload time
"""

from functools import lru_cache

from loguru import logger
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ServerSelectionTimeoutError

from config import get_settings


@lru_cache(maxsize=1)
def _get_client() -> MongoClient:
    """Return a cached MongoClient. Connection is lazy — established on first use."""
    settings = get_settings()
    return MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5_000)


def get_database():
    """Return the application database handle."""
    settings = get_settings()
    return _get_client()[settings.mongodb_database]


def raw_documents_collection() -> Collection:
    """Return the raw_documents collection handle."""
    return get_database()["raw_documents"]


def ping() -> bool:
    """Return True if MongoDB is reachable, False otherwise."""
    try:
        _get_client().admin.command("ping")
        return True
    except ServerSelectionTimeoutError as exc:
        logger.warning("MongoDB unreachable: %s", exc)
        return False


def ensure_indexes() -> None:
    """Create indexes on startup so queries stay fast as the collection grows."""
    col = raw_documents_collection()
    col.create_index("user_id")
    col.create_index([("user_id", 1), ("source", 1)])
    logger.info("MongoDB indexes ensured on raw_documents.")
