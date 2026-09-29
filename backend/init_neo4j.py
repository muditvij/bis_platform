"""
Initialize Neo4j AuraDB schema, indexes, constraints, and ingest the BIS Knowledge Base.
"""

import os
import json
import logging
from neo4j import GraphDatabase
from fastembed import TextEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(ROOT_DIR, "data", "knowledge_base.json")

# Load .env variables
ENV_PATH = os.path.join(ROOT_DIR, ".env")
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Domain specific relational data to enrich BIS Standards
SCHEMES_MAP = {
    "isi": {"id": "SCH-ISI", "name": "Scheme-I (ISI Mark)", "type": "Product Certification", "features": "Mandatory factory audit, in-house laboratory, batch inspection & third-party sample testing"},
    "crs": {"id": "SCH-CRS", "name": "Scheme-II (CRS)", "type": "Compulsory Registration", "features": "Self-declaration of conformity based on test report from BIS-recognised lab (no factory audit)"},
    "fmcs": {"id": "SCH-FMCS", "name": "Scheme-X (FMCS)", "type": "Foreign Manufacturers Certification", "features": "Certification for foreign manufacturers exporting goods to India"},
    "hallmarking": {"id": "SCH-HALLMARK", "name": "Hallmarking Scheme", "type": "Precious Metals Certification", "features": "Assaying & 6-digit alphanumeric HUID marking for gold and silver articles"},
    "schemex": {"id": "SCH-X", "name": "Scheme-X", "type": "Capital Goods & Machinery", "features": "Type testing and factory quality management surveillance for low-risk machinery"},
    "ecomark": {"id": "SCH-ECO", "name": "ECO Mark Scheme", "type": "Environmental Certification", "features": "Dual certification with ISI mark and earthen pot (Matka) logo for eco-friendly goods"}
}

QCO_MAP = {
    "helmet": {"id": "QCO-HELMET", "title": "Two Wheeler Helmets (Quality Control) Order, 2020", "ministry": "Ministry of Road Transport & Highways (MoRTH)", "status": "Mandatory", "effective_date": "2021-06-01"},
    "steel": {"id": "QCO-STEEL", "title": "Steel and Steel Products (Quality Control) Order", "ministry": "Ministry of Steel", "status": "Mandatory", "effective_date": "2020-08-01"},
    "toy": {"id": "QCO-TOYS", "title": "Toys (Quality Control) Order, 2020", "ministry": "DPIIT", "status": "Mandatory", "effective_date": "2021-01-01"},
    "footwear": {"id": "QCO-FOOTWEAR", "title": "Footwear Made from Leather and Other Materials (QCO), 2024", "ministry": "DPIIT", "status": "Mandatory", "effective_date": "2024-01-01"},
    "gold": {"id": "QCO-GOLD", "title": "Hallmarking of Gold Jewellery and Gold Artefacts Order", "ministry": "Department of Consumer Affairs", "status": "Mandatory in 343+ Districts", "effective_date": "2021-06-23"},
    "silver": {"id": "QCO-SILVER", "title": "Silver Artefacts Hallmarking Order", "ministry": "Department of Consumer Affairs", "status": "Voluntary", "effective_date": "Active"},
    "cylinder": {"id": "QCO-LPG", "title": "Gas Cylinders Rules & Mandatory ISI Certification", "ministry": "PESO / DPIIT", "status": "Mandatory", "effective_date": "Active"},
    "water": {"id": "QCO-WATER", "title": "Food Safety and Standards (Packaged Drinking Water) Mandatory Order", "ministry": "FSSAI / BIS", "status": "Mandatory", "effective_date": "Active"},
    "fire": {"id": "QCO-FIRE", "title": "Fire Fighting Equipment Quality Order", "ministry": "Ministry of Commerce & Industry", "status": "Mandatory", "effective_date": "Active"},
    "cable": {"id": "QCO-CABLES", "title": "Electrical Wires and Cables (QCO), 2023", "ministry": "DPIIT", "status": "Mandatory", "effective_date": "2023-09-01"},
    "battery": {"id": "QCO-BATTERIES", "title": "Batteries (Management and Handling) & Safety Order", "ministry": "MeitY / MoEFCC", "status": "Mandatory under CRS", "effective_date": "Active"},
    "led": {"id": "QCO-LED", "title": "Electronics and IT Goods (Compulsory Registration) Order for LEDs", "ministry": "MeitY", "status": "Mandatory under CRS", "effective_date": "Active"}
}

LABORATORIES = [
    {"id": "LAB-BIS-CL", "name": "BIS Central Laboratory (Ghaziabad)", "state": "Uttar Pradesh", "city": "Ghaziabad", "capabilities": ["Steel", "Cables", "Helmets", "Packaged Water", "Cement"]},
    {"id": "LAB-NTH-KOL", "name": "National Test House (Eastern Region)", "state": "West Bengal", "city": "Kolkata", "capabilities": ["Steel", "Chemicals", "Paints", "Cement", "Cables"]},
    {"id": "LAB-NTH-MUM", "name": "National Test House (Western Region)", "state": "Maharashtra", "city": "Mumbai", "capabilities": ["Mechanical", "Civil", "Electrical", "Rubber", "Textiles"]},
    {"id": "LAB-NTH-CHE", "name": "National Test House (Southern Region)", "state": "Tamil Nadu", "city": "Chennai", "capabilities": ["Electrical", "Electronics", "Mechanical"]},
    {"id": "LAB-CIPET-AHM", "name": "Central Institute of Petrochemicals Engg & Tech (CIPET)", "state": "Gujarat", "city": "Ahmedabad", "capabilities": ["Helmets", "Plastics", "Polymers", "Toys"]},
    {"id": "LAB-CIPET-CHE", "name": "CIPET Polymer Research Centre", "state": "Tamil Nadu", "city": "Chennai", "capabilities": ["Plastics", "Polymer Pipes", "Footwear Polymers"]},
    {"id": "LAB-NML-JAM", "name": "CSIR - National Metallurgical Laboratory", "state": "Jharkhand", "city": "Jamshedpur", "capabilities": ["Steel Bars", "Alloys", "Corrosion Resistance"]},
    {"id": "LAB-NCB-DEL", "name": "National Council for Cement and Building Materials (NCB)", "state": "Haryana", "city": "Ballabgarh", "capabilities": ["Cement", "Concrete", "Aggregates"]},
    {"id": "LAB-ARAI-PUN", "name": "Automotive Research Association of India (ARAI)", "state": "Maharashtra", "city": "Pune", "capabilities": ["Two Wheeler Helmets", "Automotive Components", "EV Batteries"]},
    {"id": "LAB-FDDI-NOI", "name": "Footwear Design & Development Institute (FDDI)", "state": "Uttar Pradesh", "city": "Noida", "capabilities": ["Safety Footwear", "Leather Products", "Chappals & Sandals"]}
]

KEY_CLAUSES = {
    "IS 4151": [
        {"id": "IS-4151-CL-7.2", "clause": "Clause 7.2", "title": "Impact Absorption Test", "desc": "Measures deceleration imparted to a headform dropped inside the helmet onto rigid anvils."},
        {"id": "IS-4151-CL-7.3", "clause": "Clause 7.3", "title": "Rigidity and Deformation Test", "desc": "Evaluates lateral deformation when compressed between two parallel plates under 630 N load."},
        {"id": "IS-4151-CL-7.4", "clause": "Clause 7.4", "title": "Retention System and Dynamic Test", "desc": "Tests strap elongation and buckle slip resistance under sudden shock loads."}
    ],
    "IS 1786": [
        {"id": "IS-1786-CL-8.1", "clause": "Clause 8.1", "title": "Tensile Strength and 0.2% Proof Stress", "desc": "Verification of yield strength (Fe 415, Fe 500, Fe 550, Fe 600 grades) and total elongation at max force."},
        {"id": "IS-1786-CL-8.2", "clause": "Clause 8.2", "title": "Bend and Rebend Test", "desc": "Ensures bars bend through specified angles around mandrels without surface rupture or transverse cracks."},
        {"id": "IS-1786-CL-4.2", "clause": "Clause 4.2", "title": "Chemical Composition Limits", "desc": "Strict maximum limits on Carbon (0.25%-0.30%), Sulphur (0.040%-0.060%), and Phosphorus."}
    ],
    "IS 1417": [
        {"id": "IS-1417-CL-4", "clause": "Clause 4", "title": "Fineness and Purity Grades", "desc": "Recognised purity grades for gold jewellery: 24K (999), 23K (958), 22K (916), 20K (833), 18K (750), 14K (585)."},
        {"id": "IS-1417-CL-6", "clause": "Clause 6", "title": "HUID and Hallmark Symbols", "desc": "Mandates BIS logo, purity/fineness mark, and 6-digit alphanumeric unique identification code (HUID)."}
    ],
    "IS 15298 (Part 2)": [
        {"id": "IS-15298-CL-5.3", "clause": "Clause 5.3", "title": "Toe Impact Resistance Test", "desc": "Steel or composite toe cap must withstand 200 Joules impact energy without collapsing."},
        {"id": "IS-15298-CL-5.4", "clause": "Clause 5.4", "title": "Compression Resistance Test", "desc": "Toe cap must withstand 15 kN compressive force maintaining required clearance."}
    ],
    "IS 14543": [
        {"id": "IS-14543-CL-4.1", "clause": "Clause 4.1", "title": "Microbiological Safety Limits", "desc": "Zero tolerance for E. coli, coliform bacteria, fecal streptococci, and Pseudomonas aeruginosa in 250 ml."},
        {"id": "IS-14543-CL-4.3", "clause": "Clause 4.3", "title": "Pesticide Residue Limits", "desc": "Individual pesticide residues not to exceed 0.0001 mg/l and total pesticides not to exceed 0.0005 mg/l."}
    ],
    "IS 9873 (Part 1)": [
        {"id": "IS-9873-CL-4.1", "clause": "Clause 4.1", "title": "Small Parts Choking Hazard", "desc": "Toys for children under 36 months must not fit into the small parts cylinder under 50 N pull force."},
        {"id": "IS-9873-CL-4.7", "clause": "Clause 4.7", "title": "Sharp Edges and Points", "desc": "Verification that accessible edges are smooth, rolled, or shielded to prevent laceration."}
    ]
}

def main():
    logger.info("Initializing Neo4j Graph Database at %s...", NEO4J_URI)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    logger.info("Connected to Neo4j successfully!")

    # 1. Initialize FastEmbed ONNX
    logger.info("Loading FastEmbed BAAI/bge-small-en-v1.5 model...")
    embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")

    with driver.session() as session:
        # 2. Constraints
        logger.info("Creating constraints...")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Standard) REQUIRE s.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (sc:Scheme) REQUIRE sc.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (q:QCO) REQUIRE q.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Laboratory) REQUIRE l.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Clause) REQUIRE c.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (k:Chunk) REQUIRE k.id IS UNIQUE")

        # 3. Vector Index for 384-dimensional embeddings on Chunk(embedding)
        logger.info("Ensuring Vector and Full-text indexes exist...")
        try:
            session.run("""
            CREATE VECTOR INDEX chunk_vector_index IF NOT EXISTS
            FOR (m:Chunk) ON (m.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """)
            logger.info("Vector index 'chunk_vector_index' created or verified.")
        except Exception as e:
            logger.warning("Vector index setup message: %s", e)

        # 4. Full-text index on Standard(is_number, title, keywords_str)
        try:
            session.run("""
            CREATE FULLTEXT INDEX standard_fulltext_index IF NOT EXISTS
            FOR (s:Standard) ON EACH [s.is_number, s.title, s.keywords_str]
            """)
            logger.info("Full-text index 'standard_fulltext_index' created or verified.")
        except Exception as e:
            logger.warning("Full-text index setup message: %s", e)

        # 5. Ingest Schemes
        logger.info("Seeding Schemes...")
        for k, sc in SCHEMES_MAP.items():
            session.run("""
            MERGE (s:Scheme {id: $id})
            SET s.name = $name,
                s.type = $type,
                s.features = $features
            """, id=sc["id"], name=sc["name"], type=sc["type"], features=sc["features"])

        # 6. Ingest QCOs
        logger.info("Seeding QCOs...")
        for k, q in QCO_MAP.items():
            session.run("""
            MERGE (node:QCO {id: $id})
            SET node.title = $title,
                node.ministry = $ministry,
                node.status = $status,
                node.effective_date = $effective_date
            """, id=q["id"], title=q["title"], ministry=q["ministry"], status=q["status"], effective_date=q["effective_date"])

        # 7. Ingest Laboratories
        logger.info("Seeding Testing Laboratories...")
        for lab in LABORATORIES:
            session.run("""
            MERGE (l:Laboratory {id: $id})
            SET l.name = $name,
                l.state = $state,
                l.city = $city,
                l.capabilities = $capabilities
            """, id=lab["id"], name=lab["name"], state=lab["state"], city=lab["city"], capabilities=lab["capabilities"])

        # 8. Load and Ingest Knowledge Base
        logger.info("Loading knowledge base from %s...", KB_PATH)
        with open(KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", [])
        logger.info("Processing %d knowledge base entries...", len(entries))

        chunk_texts = []
        chunk_metas = []

        for entry in entries:
            e_id = entry["id"]
            topic = entry.get("topic", "standard")
            is_num = entry.get("is_number") or ""
            title = entry.get("title", "")
            title_hi = entry.get("title_hi", "")
            summary = entry.get("summary", "")
            summary_hi = entry.get("summary_hi", "")
            keywords = entry.get("keywords", [])
            keywords_str = " ".join(keywords)
            next_steps = entry.get("next_steps", [])
            source_title = entry.get("source_title", "BIS Portal")
            source_url = entry.get("source_url", "https://www.services.bis.gov.in/")
            verified = entry.get("verified", True)

            # Insert Standard / Topic Node
            session.run("""
            MERGE (s:Standard {id: $id})
            SET s.is_number = $is_num,
                s.title = $title,
                s.title_hi = $title_hi,
                s.topic = $topic,
                s.summary = $summary,
                s.summary_hi = $summary_hi,
                s.keywords_str = $keywords_str,
                s.source_title = $source_title,
                s.source_url = $source_url,
                s.verified = $verified,
                s.next_steps = $next_steps
            """, id=e_id, is_num=is_num, title=title, title_hi=title_hi, topic=topic,
                 summary=summary, summary_hi=summary_hi, keywords_str=keywords_str,
                 source_title=source_title, source_url=source_url, verified=verified,
                 next_steps=next_steps)

            # Relate to Schemes
            if "SCH-ISI" in e_id or topic == "standard" and "crs" not in keywords:
                session.run("MATCH (s:Standard {id: $id}), (sc:Scheme {id: 'SCH-ISI'}) MERGE (s)-[:CERTIFIED_UNDER]->(sc)", id=e_id)
            if "SCH-CRS" in e_id or any(w in keywords for w in ["crs", "led", "mobile", "laptop", "power bank", "solar"]):
                session.run("MATCH (s:Standard {id: $id}), (sc:Scheme {id: 'SCH-CRS'}) MERGE (s)-[:CERTIFIED_UNDER]->(sc)", id=e_id)
            if any(w in keywords for w in ["gold", "silver", "hallmark", "huid"]):
                session.run("MATCH (s:Standard {id: $id}), (sc:Scheme {id: 'SCH-HALLMARK'}) MERGE (s)-[:CERTIFIED_UNDER]->(sc)", id=e_id)

            # Relate to QCOs
            for q_key, q_val in QCO_MAP.items():
                if q_key in keywords or q_key in title.lower():
                    session.run("""
                    MATCH (s:Standard {id: $std_id}), (q:QCO {id: $qco_id})
                    MERGE (s)-[:MANDATED_BY]->(q)
                    """, std_id=e_id, qco_id=q_val["id"])

            # Relate to Product Category
            for kw in keywords[:4]:
                if len(kw) > 3 and not kw.startswith("is ") and not any(ch in kw for ch in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
                    p_id = f"PROD-{kw.upper().replace(' ', '-')}"
                    session.run("""
                    MERGE (p:Product {id: $p_id})
                    ON CREATE SET p.name = $name
                    WITH p
                    MATCH (s:Standard {id: $std_id})
                    MERGE (p)-[:APPLIES_STANDARD]->(s)
                    """, p_id=p_id, name=kw.title(), std_id=e_id)

            # Prepare chunk text for vector embedding
            full_chunk_text = f"{is_num} {title}. {summary} Keywords: {keywords_str}"
            chunk_texts.append(full_chunk_text)
            chunk_metas.append({"chunk_id": f"CHK-{e_id}", "std_id": e_id, "text": full_chunk_text})

        # 9. Ingest Specific Clauses & Lab Capabilities
        logger.info("Adding Clause-level details & connecting testing laboratories...")
        for is_code, clauses in KEY_CLAUSES.items():
            for cl in clauses:
                cl_id = cl["id"]
                session.run("""
                MERGE (c:Clause {id: $id})
                SET c.clause_number = $clause,
                    c.title = $title,
                    c.description = $desc
                WITH c
                MATCH (s:Standard) WHERE s.is_number = $is_code
                MERGE (s)-[:HAS_CLAUSE]->(c)
                """, id=cl_id, clause=cl["clause"], title=cl["title"], desc=cl["desc"], is_code=is_code)

                # Connect clause to relevant testing laboratories
                for lab in LABORATORIES:
                    matched = False
                    for cap in lab["capabilities"]:
                        if cap.lower() in cl["title"].lower() or cap.lower() in cl["desc"].lower() or is_code.lower() in cap.lower():
                            matched = True
                            break
                    if matched or ("helmet" in is_code.lower() and "cipet" in lab["id"].lower()) or ("steel" in is_code.lower() and "nml" in lab["id"].lower()):
                        session.run("""
                        MATCH (c:Clause {id: $cl_id}), (l:Laboratory {id: $lab_id})
                        MERGE (c)-[:TESTED_AT]->(l)
                        """, cl_id=cl_id, lab_id=lab["id"])

        # 10. Compute and Store FastEmbed Vector Embeddings
        logger.info("Generating FastEmbed embeddings for %d chunks...", len(chunk_texts))
        embeddings = list(embed_model.embed(chunk_texts))
        logger.info("Writing Chunk nodes with embeddings into Neo4j...")
        for meta, emb in zip(chunk_metas, embeddings):
            emb_list = emb.tolist()
            session.run("""
            MERGE (k:Chunk {id: $chunk_id})
            SET k.text = $text,
                k.embedding = $embedding
            WITH k
            MATCH (s:Standard {id: $std_id})
            MERGE (s)-[:HAS_CHUNK]->(k)
            """, chunk_id=meta["chunk_id"], text=meta["text"], embedding=emb_list, std_id=meta["std_id"])

        # 11. Connect Normative References
        session.run("""
        MATCH (s1:Standard), (s2:Standard)
        WHERE s1.is_number IS NOT NULL AND s2.is_number IS NOT NULL AND s1.id <> s2.id
          AND (s1.summary CONTAINS s2.is_number OR s1.keywords_str CONTAINS s2.is_number)
        MERGE (s1)-[:REFERENCES_NORMATIVE]->(s2)
        """)

        # 12. Verification Counts
        res_nodes = session.run("MATCH (n) RETURN count(n) as total_nodes, labels(n) as label").data()
        counts_by_label = {}
        for r in res_nodes:
            lbl = r["label"][0] if r["label"] else "Unknown"
            counts_by_label[lbl] = counts_by_label.get(lbl, 0) + r["total_nodes"]

        res_edges = session.run("MATCH ()-[r]->() RETURN count(r) as total_edges, type(r) as rel_type").data()
        counts_by_rel = {}
        for r in res_edges:
            rel = r["rel_type"]
            counts_by_rel[rel] = counts_by_rel.get(rel, 0) + r["total_edges"]

        total_nodes = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
        total_edges = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]

        logger.info("INGESTION COMPLETE!")
        logger.info("Total Graph Nodes: %d", total_nodes)
        logger.info("Total Graph Edges: %d", total_edges)
        logger.info("Nodes by Label: %s", counts_by_label)
        logger.info("Edges by Relationship: %s", counts_by_rel)

    driver.close()

if __name__ == "__main__":
    main()
