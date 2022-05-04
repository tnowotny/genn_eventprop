import json
import numpy as np

## Enose spikes
model = "gas_minus_heater"
delta = 0.1
with open(f"data/events_train_{model}_{str(delta).replace('.', 'p')}.json", 'r', encoding='utf-8-sig') as fp:
    all_events_train = json.load(fp)
with open(f"data/events_test_{model}_{str(delta).replace('.', 'p')}.json", 'r', encoding='utf-8-sig') as fp:
    all_events_test = json.load(fp)

# y_train, X_train = all_events_train.keys(), all_events_train.values()
# y_test, X_test = all_events_test.keys(), all_events_test.values()
print(all_events_train.keys())


labels2num = {
    'EB':0, 
    '2H':1, 
    'IA':2, 
    'Eu':3, 
    'blank':4,
    }

y_train = []
X_train = []
for label, cycles in all_events_train.items():
    for cycle in cycles:
        sample = {"x":[], "t":[]}
        for i in range(8):
            sample["x"].extend([i for c in cycle[i]])
            sample["t"].extend([c for c in cycle[i]])

        index = np.argsort(sample["t"])
        sample["t"] = np.array(sample["t"])[index]
        sample["x"] = np.array(sample["x"])[index]
        
        y_train.append(labels2num[label])
        X_train.append(sample)

y_test = []
X_test = []
for label, cycles in all_events_test.items():
    for cycle in cycles:
        sample = {"x":[], "t":[]}
        for i in range(8):
            sample["x"].extend([i for c in cycle[i]])
            sample["t"].extend([c for c in cycle[i]])

        index = np.argsort(sample["t"])
        sample["t"] = np.array(sample["t"])[index]
        sample["x"] = np.array(sample["x"])[index]
        
        y_test.append(labels2num[label])
        X_test.append(sample)
        
print(len(y_train))
print(len(X_train))        
print(len(y_test))
print(len(X_test))