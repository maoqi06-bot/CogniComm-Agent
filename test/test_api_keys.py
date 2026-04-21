"""Quick test for API Keys validity"""

import os
import sys
from pathlib import Path

# Load .env file
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

# Test DeepSeek API
def test_deepseek():
    print("\n[Test DeepSeek API]")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("  [MISSING] DEEPSEEK_API_KEY not set")
        return False

    print(f"  API Key: {api_key[:10]}...{api_key[-4:]}")

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1/"
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10
        )
        print(f"  [OK] DeepSeek API Valid!")
        print(f"  Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"  [FAIL] DeepSeek API Failed: {e}")
        return False

# Test OpenAI API
def test_openai():
    print("\n[Test OpenAI API]")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  [MISSING] OPENAI_API_KEY not set")
        return False

    print(f"  API Key: {api_key[:10]}...{api_key[-4:]}")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10
        )
        print(f"  [OK] OpenAI API Valid!")
        print(f"  Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"  [FAIL] OpenAI API Failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("API Key Validity Test")
    print("=" * 50)

    deepseek_ok = test_deepseek()
    openai_ok = test_openai()

    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  DeepSeek: {'OK' if deepseek_ok else 'FAIL'}")
    print(f"  OpenAI:   {'OK' if openai_ok else 'FAIL'}")
    print("=" * 50)
