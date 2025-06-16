

import json

def textCleaner(input_path='input.json', output_path='output.json'):
    print("Avvio la pulizia del testo...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if 'llama4_correction' in item and isinstance(item['llama4_correction'], str):
            item['llama4_correction'] = item['llama4_correction'].replace("Ecco il testo corretto:\n\n", "")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Pulizia completata. File salvato come '{output_path}'")



