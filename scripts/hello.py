# from openai import OpenAI
# # Set OpenAI's API key and API base to use SGLang's API server.
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:30000/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# chat_response = client.chat.completions.create(
#     model="Qwen/Qwen3.5-4B",
#     messages=[
#         {"role": "user", "content": "hello, intro yourself"},
#     ],
#     max_tokens=1024,
#     temperature=0.7,
#     top_p=0.8,
#     presence_penalty=1.5,
#     # extra_body={
#     #     "top_k": 20,
#     #     "chat_template_kwargs": {"enable_thinking": False},
#     # },
# )


from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[{"role": "user", "content": "Introduce yourself."}],
    max_tokens=1024,
    temperature=0.0,
    presence_penalty=1.5,
)
print(response.choices[0].message.content)


# print("Chat response:", chat_response)