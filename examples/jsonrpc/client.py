"""JSON-RPC client example.

Run the server first::

    blaze serve examples.jsonrpc.app:app --transport jsonrpc --http-port 8080

Then run this client::

    python examples/jsonrpc/client.py
"""

import asyncio

from blazerpc import JsonRpcClient


async def main() -> None:
    async with JsonRpcClient("http://127.0.0.1:8080/jsonrpc") as client:
        # Unary calls
        echo_result = await client.predict("echo", text="hello world")
        print(f"echo  → {echo_result}")

        add_result = await client.predict("add", a=3.0, b=4.0)
        print(f"add   → {add_result}")

        greet_result = await client.predict("greet", name="Alice")
        print(f"greet → {greet_result}")

        # Streaming call
        print("words → ", end="")
        async for word in client.stream("words", sentence="hello blazerpc world"):
            print(word, end=" ")
        print()


if __name__ == "__main__":
    asyncio.run(main())
