// Routes Pi to the local kvcompact server and swaps Pi's compaction for one of three arms:
//   KVCOMPACT_MODE=kv       the server's advisor edits the KV cache; Pi gets a sentinel instead of a summary
//   KVCOMPACT_MODE=summary  Pi's own compaction prompt, but written by the advisor model (same advisor as kv)
//   KVCOMPACT_MODE=none     no compaction: the window is opened up to the model's native length
// The compaction trigger is Pi's: contextTokens > contextWindow - compaction.reserveTokens (.pi/settings.json).

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { compact } from "@mariozechner/pi-coding-agent";

const SERVER = process.env.KVCOMPACT_URL ?? "http://127.0.0.1:8000";
const MODE = process.env.KVCOMPACT_MODE ?? "kv";
const LIMIT = Number(process.env.KVCOMPACT_LIMIT ?? 32768);
const RESERVE = 8192; // keep equal to compaction.reserveTokens in .pi/settings.json
const ADVISOR = process.env.KVCOMPACT_ADVISOR ?? "claude-opus-5-5";

if (!["kv", "summary", "none"].includes(MODE)) throw new Error(`KVCOMPACT_MODE must be kv, summary or none, not ${MODE}`);

export default function (pi: ExtensionAPI) {
	pi.registerProvider("kvcompact", {
		baseUrl: `${SERVER}/v1`,
		apiKey: "local",
		api: "openai-completions",
		models: [
			{
				id: "qwen3-coder",
				name: `Qwen3-Coder (kvcompact, ${MODE})`,
				reasoning: false,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: MODE === "none" ? 262144 : LIMIT + RESERVE,
				maxTokens: 4096,
				compat: { supportsDeveloperRole: false, supportsReasoningEffort: false, maxTokensField: "max_tokens" },
			},
		],
	});

	pi.on("session_before_compact", async (event, ctx) => {
		const { preparation, customInstructions, signal } = event;

		if (MODE === "none") return { cancel: true };

		if (MODE === "summary") {
			// Same rule as kv: never let a failure fall through to Pi's default (Qwen-written) summary.
			const model = ctx.modelRegistry.find("anthropic", ADVISOR);
			const auth = model ? await ctx.modelRegistry.getApiKeyAndHeaders(model) : undefined;
			if (!model || !auth?.ok || !auth.apiKey) {
				console.error(`[kvcompact] advisor ${ADVISOR} unavailable (log in to anthropic in pi or set ANTHROPIC_API_KEY), compaction cancelled`);
				return { cancel: true };
			}
			try {
				return { compaction: await compact(preparation, model, auth.apiKey, auth.headers, customInstructions, signal) };
			} catch (error) {
				console.error(`[kvcompact] summary compaction FAILED, compaction cancelled: ${error}`);
				return { cancel: true };
			}
		}

		// kv: the server compacts the cache it holds and tells us which compaction this is.
		// If this handler throws, Pi falls back to its DEFAULT summary compaction (seen in testing), which would
		// silently turn a kv run into a summary run. So failures cancel instead, and say so on stderr.
		// Don't touch ctx after an await: in print mode it can be stale by then and throws.
		let result: { id: string; slots: number; logical: number };
		try {
			const response = await fetch(`${SERVER}/compact`, { method: "POST", signal });
			if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
			result = (await response.json()) as typeof result;
		} catch (error) {
			console.error(`[kvcompact] kv compaction FAILED, compaction cancelled: ${error}`);
			return { cancel: true };
		}
		const { id, slots, logical } = result;
		console.error(`[kvcompact] kv-compacted ${logical} tokens into ${slots} slots (${id})`);
		return {
			compaction: {
				summary: `[kv-compacted:${id}]`,
				firstKeptEntryId: preparation.firstKeptEntryId,
				tokensBefore: preparation.tokensBefore,
				details: { id, slots, logical },
			},
		};
	});
}
