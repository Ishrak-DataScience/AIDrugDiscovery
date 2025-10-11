import requests

# URL for RCSB Search API
url = "https://search.rcsb.org/rcsbsearch/v2/query?json="

# Query for all entries (sorted by release date)
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
        "paginate": {
            "start": 0,
            "rows": 10000
        },
        "results_content_type": ["experimental"],
        "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}]
    },
    "return_type": "entry"
}

response = requests.post(url, json=query)
data = response.json()

# Extract PDB IDs
pdb_ids = [result['identifier'] for result in data.get('result_set', [])]

print(f"Fetched {len(pdb_ids)} PDB IDs:")
print(pdb_ids[:20])  # show first 20

with open("pdb_ids.txt", "w") as f:
    for pid in pdb_ids:
        f.write(pid + "\n")

print("Saved PDB IDs to pdb_ids.txt")