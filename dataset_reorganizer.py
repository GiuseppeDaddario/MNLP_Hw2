def parse_icdar_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError("File troppo corto, aspettate 3 righe con i dati")

    data = {}
    for line in lines[:3]:
        line_clean = line.lstrip()
        parts = line_clean.split(" ", 1)
        if len(parts) < 2:
            raise ValueError(f"Formato riga non valido: {repr(line)}")
        tag, text = parts
        tag = tag.strip().lower()
        if tag == "[ocr_toinput]":
            data["ocr_raw"] = text
        elif tag == "[ocr_aligned]":
            data["ocr_aligned"] = text
        elif tag == "[gs_aligned]" or tag == "[ gs_aligned]":
            data["gs_aligned"] = text
        else:
            raise ValueError(f"Tag non riconosciuto: {repr(tag)}")

    return data

# Esempio d'uso
filepath = "datasets/eng_monograph/666.txt"
parsed = parse_icdar_file(filepath)
print("OCR Raw:", parsed["ocr_raw"][:100])        # Stampa i primi 100 caratteri
print("OCR Aligned:", parsed["ocr_aligned"][:100])
print("GS Aligned:", parsed["gs_aligned"][:100])
