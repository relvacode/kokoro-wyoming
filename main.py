#!/usr/bin/env python3
import argparse
import asyncio
import logging
import json
from dataclasses import dataclass
from typing import Optional
from functools import partial
from wyoming.server import AsyncEventHandler
from kokoro import KPipeline
import numpy as np

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker, Describe, Info
from wyoming.server import AsyncServer
from wyoming.tts import Synthesize, SynthesizeVoice
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event

_LOGGER = logging.getLogger(__name__)
VERSION = "0.1"

@dataclass
class KokoroVoice:
    name: str
    language: str
    kokoro_id: str

# Define available voices
VOICES = [
            "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael",
            "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis"
        ]

voices = [
            TtsVoice(
                name=voice_id,
                description=f"Kokoro voice {voice_id}",
                attribution=Attribution(
                    name="Kokoro", url="https://github.com/CjangCjengh/kokoro"
                ),
                installed=True,
                version=None,
                languages=[
                    voice_id.split("_")[0]
                ],
                speakers=[
                    TtsVoiceSpeaker(name=voice_id.split("_")[1])
                ]
            )
            for voice_id in VOICES
        ]

class KokoroEventHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info,
        cli_args: argparse.Namespace,
        *args,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.args = args
        self.pipeline = KPipeline(lang_code='a')  # Initialize with English
        self.wyoming_info_event = wyoming_info.event()

    async def handle_event(self, event: Event):
        """Handle Wyoming protocol events."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if not Synthesize.is_type(event.type):
            _LOGGER.warning("Unexpected event: %s", event)
            return True

        try:
            return await self._handle_synthesize(event)
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def handle_client(
        self, 
        client_reader: asyncio.StreamReader, 
        client_writer: asyncio.StreamWriter
    ):
        try:
            while True:
                event = await AsyncServer.read_event(client_reader)
                if event is None:
                    break

                if isinstance(event, Synthesize):
                    await self._handle_synthesize(event, client_writer)
                    break

        except Exception as e:
            _LOGGER.exception("Error handling client: %s", e)
        finally:
            client_writer.close()
            await client_writer.wait_closed()

    """Handle text to speech synthesis request."""
    async def _handle_synthesize(self, event: Event): 
        try:
            synthesize = Synthesize.from_event(event)

            print("Got synthesize event!")
            print(synthesize)
            # Get voice settings
            voice_name = "american-bella"  # default voice
            if synthesize.voice:
                voice_name = synthesize.voice.name

            # Find matching voice
            voice = next((v for v in VOICES if v == voice_name), VOICES[0])

            # Generate audio
            generator = self.pipeline(
                synthesize.text,
                voice=voice.kokoro_id,
                speed=1
            )

            # Send audio start
            await AsyncServer.write_event(
                AudioStart(
                    rate=24000,
                    width=2,
                    channels=1,
                ),
                client_writer,
            )

            # Process each chunk
            for _, _, audio in generator:
                # Convert float32 to int16
                audio_int16 = (audio * 32767).astype(np.int16)
                
                # Send audio chunk
                await AsyncServer.write_event(
                    AudioChunk(
                        audio=audio_int16.tobytes(),
                        rate=24000,
                        width=2,
                        channels=1,
                    ),
                    client_writer,
                )

            # Send audio stop
            await AsyncServer.write_event(
                AudioStop(),
                client_writer,
            )

        except Exception as e:
            _LOGGER.exception("Error synthesizing: %s", e)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to listen on"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=10200, 
        help="Port to listen on"
    )
    parser.add_argument(
        "--uri",
        default="tcp://0.0.0.0:10210",
        help="unix:// or tcp://"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    wyoming_info = Info(
            tts=[TtsProgram(
                name="kokoro",
                description="A fast, local, kokoro-based tts engine",
                attribution=Attribution(
                    name="Kokoro TTS",
                    url="https://huggingface.co/hexgrad/Kokoro-82M",
                ),
                installed=True,
                voices=sorted(voices, key=lambda v: v.name),
                version=VERSION
            )]
        )

    server = AsyncServer.from_uri(args.uri)

    # Start server
    await server.run(partial(KokoroEventHandler, wyoming_info, args))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass