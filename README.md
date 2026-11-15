# Free Model Aggregator

One server that serves an OpenAI- and an Anthropic-compatible API over ten
simulated free-tier model providers, routing each of 20 requests to the best
free model able to serve it, failing over when a provider refuses, and
reporting the catalog it read.

Built for the make.fun bounty
[free-model-aggregator-01](https://make.fun/bounty/free-model-aggregator-01)
("The best free AI model aggregator").

## How it works

- Reads each provider's catalog (its own auth header, its own model list) and
  registers one catalog document.
- Plans routing with a min-cost flow: every request goes to the slot with the
  lowest expected cost — rating deficit plus a risk term from the exact
  refusal distribution — re-planned before each request.
- Translates across the three wires (OpenAI, Anthropic, Gemini) with full
  echo fidelity: system/user text, tools, `max_tokens`, stop reasons, usage
  numbers, images, and JSON mode are forwarded exactly as the client sent
  them.
- Fails over on 402/429 refusals (two-strikes: a provider is dropped only on
  its second refusal, since a first refusal is usually a one-off).
- Deterministic given the brief, the requests, and the providers' answers:
  no wall clock, no unseeded randomness, so a replay leaves the same state
  export.

## Run

Python 3.13, standard library only — nothing to install:

```sh
python3 main.py
```

Print the catalog document the server registers, byte for byte:

```sh
python3 main.py --list
```

## License

MIT — see [LICENSE](LICENSE).
