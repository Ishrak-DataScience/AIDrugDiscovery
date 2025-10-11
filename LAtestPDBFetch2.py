import requests
import pandas as pd
import time

def fetch_pdb_and_ligands(n=1000, batch_size=200):
    url = "https://search.rcsb.org/rcsbsearch/v2/query?json="
    all_entries = []
    start = 0

    while len(all_entries) < n:
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
                "paginate": {"start": start, "rows": batch_size},
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

        response = requests.post(url, json=query)
        response.raise_for_status()
        data = response.json()
        batch = data.get("result_set", [])
        if not batch:
            break
        all_entries.extend(batch)
        start += len(batch)
        time.sleep(0.2)

    pdb_ligands = []
    for item in all_entries:
        pdb_id = item.get("identifier")
        fields = item.get("services", [{}])[0].get("result_fields", {})
        ligs = None

        # Try nested or flat ligand field
        if isinstance(fields, dict):
            info = fields.get("rcsb_entry_info")
            if isinstance(info, dict):
                ligs = info.get("nonpolymer_bound_components")
            if ligs is None:
                ligs = fields.get("rcsb_entry_info.nonpolymer_bound_components")

        lig_ids = []
        if isinstance(ligs, list):
            for l in ligs:
                if isinstance(l, dict) and "comp_id" in l:
                    lig_ids.append(l["comp_id"])
                elif isinstance(l, str):
                    lig_ids.append(l)

        for lig in lig_ids:
            pdb_ligands.append({"PDB_ID": pdb_id, "ligand_resname": lig})

    return pdb_ligands


if __name__ == "__main__":
    data = fetch_pdb_and_ligands(n=1000, batch_size=200)
    df = pd.DataFrame(data)
    df.to_csv("pdb_ligand_table.csv", index=False)
    print(df.head(20))
    print(f"\nSaved to pdb_ligand_table.csv with {len(df)} rows.")