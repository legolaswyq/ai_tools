import json 
import pprint

file = "/home/walter/git/ai_tools/common/id_description.json"
with open(file, 'r') as f:
    data = json.load(f)

items = data['items']

barcodes = []
for item in items:
    barcode = item['barcode']
    barcodes.append(barcode)

print(len(barcodes))
# print(barcodes)