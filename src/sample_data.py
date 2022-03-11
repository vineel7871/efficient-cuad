import json


# create sample dataset

data_path = "../data/CUAD_v1.json"

sample_path = "../data/cuad_sample.json"

with open(data_path,"r") as fobj:
    data = json.loads(fobj.read())

data["data"] = data["data"][:2]

with open(sample_path, "w") as wobj:
    wobj.write(json.dumps(data))