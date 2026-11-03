# Fallback drill

`tools/fallback_drill.py --role answer --questions 3`, 2026-11-03, the production router
(config/agent.yaml, config/limits.yaml) against the real providers. `exhausted`: the first model's
daily quota is marked spent in a local ledger, so it is refused with no call. `unreachable`: its
connection fails. `all-down`: every connection fails and the extractive answer (quoted passages
with citations, no model) serves. The next model in the chain answered within about a second;
when nothing answers, the user still gets cited passages in under 10 ms.

Chain for this role: groq:openai/gpt-oss-20b -> groq:openai/gpt-oss-120b -> groq:qwen/qwen3.8-27b -> openrouter:google/gemma-4-31b-it:free -> openrouter:nvidia/nemotron-3-ultra-550b-a55b:free -> nvidia:openai/gpt-oss-20b -> nvidia:moonshotai/kimi-k2.6 -> nvidia:mistralai/mistral-large-4-1t-a52b -> nvidia:nvidia/nemotron-3-ultra-550b-a55b -> nvidia:deepseek-ai/deepseek-v4.1-flash -> extractive answer

| scenario | item | served by | ms | attempts that failed (all-down) |
|---|---|---|---|---|
| exhausted | exs-0001 | openai/gpt-oss-120b | 998 | - |
| exhausted | exs-0002 | openai/gpt-oss-120b | 636 | - |
| exhausted | exs-0003 | openai/gpt-oss-120b | 827 | - |
| unreachable | exs-0001 | openai/gpt-oss-120b | 699 | - |
| unreachable | exs-0002 | openai/gpt-oss-120b | 555 | - |
| unreachable | exs-0003 | openai/gpt-oss-120b | 671 | - |
| all-down | exs-0001 | extractive | 9 | groq:openai/gpt-oss-20b APIConnectionError; groq:openai/gpt-oss-120b APIConnectionError; groq:qwen/qwen3.8-27b APIConnectionError; openrouter:google/gemma-4-31b-it:free APIConnectionError; openrouter:nvidia/nemotron-3-ultra-550b-a55b:free APIConnectionError; nvidia:openai/gpt-oss-20b APIConnectionError; nvidia:moonshotai/kimi-k2.6 APIConnectionError; nvidia:mistralai/mistral-large-4-1t-a52b APIConnectionError; nvidia:nvidia/nemotron-3-ultra-550b-a55b APIConnectionError; nvidia:deepseek-ai/deepseek-v4.1-flash APIConnectionError |
| all-down | exs-0002 | extractive | 7 | groq:openai/gpt-oss-20b APIConnectionError; groq:openai/gpt-oss-120b APIConnectionError; groq:qwen/qwen3.8-27b APIConnectionError; openrouter:google/gemma-4-31b-it:free APIConnectionError; openrouter:nvidia/nemotron-3-ultra-550b-a55b:free APIConnectionError; nvidia:openai/gpt-oss-20b APIConnectionError; nvidia:moonshotai/kimi-k2.6 APIConnectionError; nvidia:mistralai/mistral-large-4-1t-a52b APIConnectionError; nvidia:nvidia/nemotron-3-ultra-550b-a55b APIConnectionError; nvidia:deepseek-ai/deepseek-v4.1-flash APIConnectionError |
| all-down | exs-0003 | extractive | 7 | groq:openai/gpt-oss-20b APIConnectionError; groq:openai/gpt-oss-120b APIConnectionError; groq:qwen/qwen3.8-27b APIConnectionError; openrouter:google/gemma-4-31b-it:free APIConnectionError; openrouter:nvidia/nemotron-3-ultra-550b-a55b:free APIConnectionError; nvidia:openai/gpt-oss-20b APIConnectionError; nvidia:moonshotai/kimi-k2.6 APIConnectionError; nvidia:mistralai/mistral-large-4-1t-a52b APIConnectionError; nvidia:nvidia/nemotron-3-ultra-550b-a55b APIConnectionError; nvidia:deepseek-ai/deepseek-v4.1-flash APIConnectionError |
