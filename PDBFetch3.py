import requests
import time
import csv
import json

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query?json="

def fetch_entries_with_ligands(n=1000, batch_size=200, sort_desc_by_date=True):
    """
    Uses the RCSB Search API to fetch up to n entries and the attribute
    rcsb_entry_info.nonpolymer_bound_components (list of ligand comp_ids).
    """
    all_results = []
    rows_fetched = 0
    start = 0

    sort_clause = [{"sort_by": "rcsb_accession_info.initial_release_date",
                    "direction": "desc" if sort_desc_by_date else "asc"}]

    while rows_fetched < n:
        rows = min(batch_size, n - rows_fetched)
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    # Return any entry that exists (broad query)
                    "operator": "exists",
                    "attribute": "rcsb_entry_container_identifiers.entry_id"
                }
            },
            "request_options": {
                "paginate": {"start": start, "rows": rows},
                "results_content_type": ["experimental"],  # omit computational models
                "sort": sort_clause,
                # ask the API to include both entry_id and ligand list
                "attributes": [
                    "rcsb_entry_container_identifiers.entry_id",
                    "rcsb_entry_info.nonpolymer_bound_components"
                ]
            },
            "return_type": "entry"
        }

        resp = requests.post(SEARCH_URL, json=query, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("result_set", [])
        if not batch:
            break  # no more results
        all_results.extend(batch)
        got = len(batch)
        rows_fetched += got
        start += got
        # be a good citizen
        time.sleep(0.2)

    # Normalize to a simple list of dicts: {entry_id, ligand_ids: [comp_ids]}
    normalized = []
    for r in all_results:
        entry_id = r.get("identifier")
        # attributes may include nested: "rcsb_entry_info": {"nonpolymer_bound_components": [...]}
        attrs = r.get("services", [{}])[0].get("result_fields", {})
        # Depending on API version, fields may be exposed top-level or nested.
        # Try both:
        ligs = None
        # 1) nested under rcsb_entry_info
        info = attrs.get("rcsb_entry_info") if isinstance(attrs, dict) else None
        if isinstance(info, dict):
            ligs = info.get("nonpolymer_bound_components")
        # 2) flat key if given directly
        if ligs is None:
            ligs = attrs.get("rcsb_entry_info.nonpolymer_bound_components")

        # Ligands should be a list of dicts with comp_id OR a list of comp_ids; normalize to comp_ids
        comp_ids = []
        if isinstance(ligs, list):
            for item in ligs:
                if isinstance(item, dict) and "comp_id" in item:
                    comp_ids.append(item["comp_id"])
                elif isinstance(item, str):
                    comp_ids.append(item)
        # Deduplicate while preserving order
        seen = set()
        uniq_comp_ids = [x for x in comp_ids if not (x in seen or seen.add(x))]

        normalized.append({
            "entry_id": entry_id,
            "ligand_ids": uniq_comp_ids
        })

    return normalized[:n]

def save_csv(data, path="pdb_ids_with_ligands.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_id", "ligand_ids"])  # ligand_ids as semicolon-separated list
        for row in data:
            ligs = ";".join(row["ligand_ids"]) if row["ligand_ids"] else ""
            writer.writerow([row["entry_id"], ligs])

def save_json(data, path="pdb_ids_with_ligands.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Fetch 1000 entries (latest first) with their ligand IDs
    results = fetch_entries_with_ligands(n=1000, batch_size=200, sort_desc_by_date=True)
    print(f"Fetched {len(results)} entries.")

    # Preview first 5
    for r in results[:5]:
        print(r["entry_id"], r["ligand_ids"])

    # Save
    save_csv(results, "pdb_ids_with_ligands.csv")
    save_json(results, "pdb_ids_with_ligands.json")
    print("Saved to pdb_ids_with_ligands.csv and pdb_ids_with_ligands.json")
