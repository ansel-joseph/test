from openai import OpenAI
import os
from dotenv import load_dotenv
from RealtimeSTT import AudioToTextRecorder


def main():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    print("API KEY loaded successfully.")

    messages = [
    {
        "role": "system",
        "content": (
            "Your name is Timothy. You are a sarcastic, witty, slightly dramatic AI assistant. "
            "You help the user efficiently but with dry humor and playful sarcasm. "
            "Never be rude, but you can tease lightly. Keep responses short and engaging. "
            "The user's name is Ansel."
        )
    }
]

    recorder = AudioToTextRecorder(model="tiny.en", language="en", spinner=False)

    try:
        while True:
            print("You: ", end="", flush=True)
            user_input = recorder.text()
            print(user_input)

            if user_input.lower() == "exit":
                print("Exiting Timothy...")
                break

            print("Timothy: ", end="", flush=True)

            messages.append({"role": "user", "content": user_input})

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                stream=True
            )

            assistant_reply = ""

            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    print(delta.content, end="", flush=True)
                    assistant_reply += delta.content

            print()
            messages.append({"role": "assistant", "content": assistant_reply})

    except Exception as e:
        print(f"\n[Error] {e}")
        print("Timothy is shutting down.")

    finally:
        print("Cleaning up...")
        recorder.shutdown()
        print("Timothy offline.")


if __name__ == "__main__":
    main()