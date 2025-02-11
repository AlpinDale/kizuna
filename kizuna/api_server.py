import io
import warnings

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from kizuna import KPipeline

# TODO: fix this warning
warnings.filterwarnings(
    "ignore", message="dropout option adds dropout after all but last recurrent layer"
)

app = FastAPI(title="Kizuna TTS API")

pipeline = KPipeline(lang_code="a")


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0
    lang_code: str = "a"
    split_pattern: str = r"\n+"


@app.post("/tts/stream")
async def text_to_speech(request: TTSRequest):
    try:
        buffer = io.BytesIO()

        generator = pipeline(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            split_pattern=request.split_pattern,
        )

        audio_chunks = []
        for _, _, audio in generator:
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            raise HTTPException(status_code=400, detail="No audio generated")

        full_audio = torch.cat(audio_chunks, dim=0)

        sf.write(buffer, full_audio.cpu().numpy(), samplerate=24000, format="WAV")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="audio.wav"'},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=2242)
