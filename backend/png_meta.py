import png

def read_text_chunks(file):
    reader = png.Reader(file)
    chunks = reader.chunks()
    text_chunks = []
    for chunk_type, chunk_data in chunks:
        if chunk_type == b'tEXt':
            text_chunks.append(chunk_data)
    return text_chunks

with open("example.png", "rb") as file:
    text_chunks = read_text_chunks(file)
    for chunk in text_chunks:
        text = chunk.decode('utf-8')
        print(text)