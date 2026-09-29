import os
from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

with driver.session() as session:
    # 1. Count existing nodes and relationships
    r = session.run("MATCH (n) RETURN count(n) as node_count").single()
    print(f"Initial node count: {r['node_count']}")
    
    # 2. Delete all existing nodes, relationships
    print("Deleting all existing nodes and relationships...")
    session.run("MATCH (n) DETACH DELETE n")
    
    # 3. Drop all existing vector and fulltext indexes and constraints if any
    try:
        indexes = session.run("SHOW INDEXES").data()
        for idx in indexes:
            idx_name = idx.get("name")
            idx_type = idx.get("type")
            # don't drop lookup indexes if system
            if idx_name and idx_type in ("VECTOR", "FULLTEXT"):
                print(f"Dropping index: {idx_name}")
                session.run(f"DROP INDEX {idx_name} IF EXISTS")
    except Exception as e:
        print(f"Note on index cleanup: {e}")

    # 4. Verify count is now 0
    r = session.run("MATCH (n) RETURN count(n) as node_count").single()
    print(f"Post-cleanup node count: {r['node_count']}")

driver.close()
print("Cleanup complete!")
