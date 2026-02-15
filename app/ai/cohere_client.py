from cohere import ClientV2
import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
co = ClientV2(api_key=os.getenv("COHERE_API_KEY"))

response = co.chat(
    model="command-a-03-2025",
    message="Hello! This is a test."
)

print(response.text)
