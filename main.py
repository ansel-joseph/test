from openai import OpenAI
import os
from dotenv import load_dotenv
from RealtimeSTT import AudioToTextRecorder
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from commands import execute_command

MAX_OUTPUT_TOKENS = 50

def main():
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")

    if not elevenlabs_api_key:
        raise ValueError("ELEVENLABS_API_KEY is not set in the environment variables.")

    print("API KEY loaded successfully.")

    client = OpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    elevenlabs = ElevenLabs(
        api_key=elevenlabs_api_key
    )

    messages = [
        {
            "role": "system",
            "content": "The user's name is Ansel"
        }
    ]

    recorder = AudioToTextRecorder(model="base.en", language="en", spinner=False)

    while True:
        print("You: ", end="", flush=True)
        user_input = recorder.text()
        print(user_input)

        if user_input.lower() == "exit.":
            break

        command_response = execute_command(user_input)

        if command_response:
            print("Timothy:", command_response)
            continue

        print("Timothy: ", end="", flush=True)

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True
        )

        full_response = ""

        for chunk in response:
            text = chunk.choices[0].delta.content or ""
            print(text, end="", flush=True)
            full_response += text

        print()

        if full_response:
            audio = elevenlabs.text_to_speech.convert(
                text=full_response,
                voice_id="bIHbv24MWmeRgasZH58o",
                model_id="eleven_flash_v2_5",
                output_format="mp3_44100_128"
            )

            play(audio)

    recorder.shutdown()


if __name__ == "__main__":
    main()