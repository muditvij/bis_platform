import os
import traceback
from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")
usernames = [os.getenv("NEO4J_USERNAME", "neo4j"), "neo4j"]
protocols = [URI]

for proto in protocols:
    for u in usernames:
        print(f"\n--- Testing uri='{proto}' with user='{u}' ---")
        try:
            driver = GraphDatabase.driver(proto, auth=(u, PASSWORD))
            driver.verify_connectivity()
            print(f"SUCCESS! Connected with uri='{proto}' user='{u}'")
            with driver.session() as session:
                res = session.run("RETURN 1 as val")
                print("Result:", [r["val"] for r in res])
            driver.close()
            exit(0)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

