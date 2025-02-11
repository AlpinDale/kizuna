# Kizuna (絆)

<p align="center">
  <img src="https://github.com/user-attachments/assets/4739007f-60d0-42a9-a952-6550d189e922" width="200"><br>
  <em>Calligraphy Artist: KANAZAWA SHOKO</em>
</p>

Kizuna is a text-to-speech (TTS) library based on [Kokoro](https://github.com/hexgrad/kokoro).
Essentially a re-write of the more expensive operations using custom Cuda kernels. Kizuna also adds a simple API server.

This project is still under progress and currently made public for collaboration purposes. Please use the original repo until this is done!

## Installation

```bash
git clone https://github.com/hexgrad/kizuna.git && cd kizuna
pip install -e .
```

## Usage

There's an example script at [example_usage.py](./examples/example_usage.py), but you can also use the REST API server.

```bash
python -m kizuna.api.server
```

You can then send a request like this:

```py
import requests
url = "http://localhost:2242/tts/stream"

data = {
    "text": "Once upon a time, there was a cat. The cat was very hungry. The cat went to the store and bought a mouse. The cat ate the mouse. The end.",
    "voice": "af_heart",
    "speed": 1.0,
    "lang_code": "a"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
```

## TODO


- [ ] Fix iSTFT kernels
- [ ] Add kernels for LeakyReLU
- [ ] Improve API server
- [ ] Add proper scheduling
