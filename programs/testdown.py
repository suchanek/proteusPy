import requests
import json

repo_url = "https://github.com/suchanek/proteusPy"
file_path = "proteusPy/data/PDB_SS_LOADER.pkl" 

# Get object ID
response = requests.get(f"{repo_url}/info/lfs/objects")
data = json.loads(response.content)

for obj in data["objects"]:
    if obj["path"] == file_path:
        object_id = obj["oid"]
        break

# Construct download URL 
download_url = f"{repo_url}/git/lfs/objects/{object_id}/blobs/{object_id}"

# Download file
response = requests.get(download_url)

with open("PDB_SS_LOADER.pkl", "wb") as f:
    f.write(response.content)

print("File downloaded!")
