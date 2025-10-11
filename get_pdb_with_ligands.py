import requests
import csv
import time

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query?json="

def fetch_pdb_with_ligands_only(n_entries=1000, batch_size=200, exclude={"HOH"}):
    pdb_ids = set()
    start = 0
    fetched = 0

    while fetched < n_entries:
        rows_this_batch = min(batch_size, n_entries - fetched)
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "operator": "exists",
                    "attribute": "rcsb_entry_container_identifiers.entry_id"
                }
            },
            "request_options": {
                "paginate": {"start": start, "rows": rows_this_batch},
                "results_content_type": ["experimental"],
                "sort": [
                    {"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}
                ],
                "attributes": [
                    "rcsb_entry_container_identifiers.entry_id",
                    "rcsb_entry_info.nonpolymer_bound_components"
                ]
            },
            "return_type": "entry"
        }

        resp = requests.post(SEARCH_URL, json=query, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("result_set", [])
        if not batch:
            break

        for rec in batch:
            pdb_id = rec.get("identifier")
            fields = rec.get("services", [{}])[0].get("result_fields", {}) or {}

            lig_list = None
            info = fields.get("rcsb_entry_info")
            if isinstance(info, dict):
                lig_list = info.get("nonpolymer_bound_components")
            if lig_list is None:
                lig_list = fields.get("rcsb_entry_info.nonpolymer_bound_components")

            comp_ids = []
            if isinstance(lig_list, list):
                for item in lig_list:
                    if isinstance(item, dict) and "comp_id" in item:
                        comp_ids.append(item["comp_id"])
                    elif isinstance(item, str):
                        comp_ids.append(item)

            comp_ids = [c for c in comp_ids if c and c not in exclude]
            if comp_ids:
                pdb_ids.add(pdb_id)

        got = len(batch)
        fetched += got
        start += got
        time.sleep(0.2)

    return sorted(pdb_ids)

if __name__ == "__main__":
    pdb_with_ligands = fetch_pdb_with_ligands_only(n_entries=1000, batch_size=200)

    with open("pdb_ids_with_ligands.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PDB_ID"])
        for pdb in pdb_with_ligands:
            writer.writerow([pdb])

    print(f"Saved {len(pdb_with_ligands)} PDB IDs with ligands to pdb_ids_with_ligands.csv")
