import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";

const EXIT_CANT_CREATE = 74;
const REGISTRATION_KEY = Symbol.for("llm-query-utils.pi-usage.registered-apis");
const USAGE_FIELDS = [
  "input",
  "output",
  "cacheRead",
  "cacheWrite",
  "cacheWrite5m",
  "cacheWrite1h",
  "reasoning",
  "totalTokens",
];
const COST_FIELDS = ["input", "output", "cacheRead", "cacheWrite", "total"];

function warn(message) {
  process.stderr.write(`pi usage: ${message}\n`);
}

function fatal(message) {
  fs.writeSync(2, `pi usage: ${message}\n`);
  process.exit(EXIT_CANT_CREATE);
}

function readConfig() {
  let config;
  try {
    config = JSON.parse(process.env.LLM_USAGE_CONFIG || "");
  } catch (_error) {
    fatal("invalid LLM_USAGE_CONFIG");
  }

  const fields = ["project_id", "profile_path", "spool_dir", "destination", "python"];
  if (
    !config ||
    typeof config !== "object" ||
    fields.some((field) => typeof config[field] !== "string" || config[field].length === 0)
  ) {
    fatal("invalid LLM_USAGE_CONFIG");
  }
  return config;
}

function selectedUsage(usage) {
  const selected = {};
  for (const field of USAGE_FIELDS) {
    if (Object.prototype.hasOwnProperty.call(usage, field)) selected[field] = usage[field];
  }
  if (usage.cost && typeof usage.cost === "object") {
    const cost = {};
    for (const field of COST_FIELDS) {
      if (Object.prototype.hasOwnProperty.call(usage.cost, field)) cost[field] = usage.cost[field];
    }
    if (Object.keys(cost).length > 0) selected.cost = cost;
  }
  return selected;
}

function timestamp(value) {
  return new Date(value).toISOString();
}

function saveRecord(config, record) {
  const spoolDir = config.spool_dir;
  const finalPath = path.join(spoolDir, `${record.id}.json`);
  const tempPath = path.join(
    spoolDir,
    `.${record.id}.${process.pid}.${randomUUID()}.tmp`,
  );
  let descriptor;
  try {
    fs.mkdirSync(spoolDir, { recursive: true, mode: 0o700 });
    fs.chmodSync(spoolDir, 0o700);
    descriptor = fs.openSync(tempPath, "wx", 0o600);
    fs.writeFileSync(descriptor, `${JSON.stringify(record)}\n`, "utf8");
    fs.fsyncSync(descriptor);
    fs.closeSync(descriptor);
    descriptor = undefined;
    fs.renameSync(tempPath, finalPath);
  } catch (error) {
    if (descriptor !== undefined) {
      try {
        fs.closeSync(descriptor);
      } catch (_closeError) {
        // The capture error below is the actionable failure.
      }
    }
    try {
      fs.unlinkSync(tempPath);
    } catch (_unlinkError) {
      // The temporary file might not exist.
    }
    fatal(`usage capture failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function baseRecord(config, ctx, values) {
  const stage = process.env.PI_SUBAGENT_AGENT || "scheduler";
  return {
    id: randomUUID(),
    tstp: timestamp(values.timestamp),
    model_name: `${values.provider}/${values.model}`,
    process_id: `${config.project_id}/${stage}${values.eventKind === "compaction" ? "/compaction" : ""}`,
    session_id: String(ctx.sessionManager.getSessionId()),
    usage: selectedUsage(values.usage),
    event_kind: values.eventKind,
    attribution: values.attribution,
    requested_model: values.requestedModel,
    job_id: process.env.PI_TMUX_SUBAGENTS_JOB_ID || null,
    parent_job_id: process.env.PI_TMUX_SUBAGENTS_PARENT_ID || null,
    destination: config.destination,
  };
}

async function flush(pi, config) {
  try {
    const result = await pi.exec(
      config.python,
      [
        "-m",
        "llm_query_utils.pi_logging",
        "_flush-worker",
        "--config",
        config.profile_path,
      ],
      { timeout: 8000 },
    );
    if (result.killed) warn("usage upload timed out; pending records retained");
    else if (result.code !== 0) warn(`usage upload failed (exit ${result.code})`);
  } catch (error) {
    warn(`usage upload failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export default function piUsageExtension(pi) {
  const config = readConfig();
  let registeredApis = globalThis[REGISTRATION_KEY];
  if (!registeredApis) {
    registeredApis = new WeakSet();
    globalThis[REGISTRATION_KEY] = registeredApis;
  }
  if (registeredApis.has(pi)) return;
  registeredApis.add(pi);

  const seenMessages = new WeakSet();
  const seenCompactionEntries = new WeakSet();
  const seenCompactionIds = new Set();
  let compactionModel = null;

  pi.on("message_end", async (event, ctx) => {
    const message = event.message;
    if (!message || message.role !== "assistant") return;
    if (seenMessages.has(message)) return;
    seenMessages.add(message);
    if (!message.usage || typeof message.usage !== "object") {
      warn("assistant usage missing; response not recorded");
      return;
    }

    const hasResponseModel = message.responseModel !== undefined && message.responseModel !== null;
    saveRecord(
      config,
      baseRecord(config, ctx, {
        timestamp: message.timestamp,
        provider: message.provider,
        model: hasResponseModel ? message.responseModel : message.model,
        requestedModel: message.model,
        usage: message.usage,
        eventKind: "response",
        attribution: hasResponseModel ? "response_model" : "request_model",
      }),
    );
  });

  pi.on("session_before_compact", async (_event, ctx) => {
    compactionModel = ctx.model
      ? { provider: ctx.model.provider, model: ctx.model.id }
      : null;
  });

  pi.on("session_compact", async (event, ctx) => {
    const entry = event.compactionEntry;
    const repeated = entry && (
      seenCompactionEntries.has(entry) ||
      (entry.id && seenCompactionIds.has(entry.id))
    );
    if (repeated) return;
    if (entry && typeof entry === "object") seenCompactionEntries.add(entry);
    if (entry && entry.id) seenCompactionIds.add(entry.id);

    if (event.fromExtension || (entry && entry.fromHook)) {
      warn("custom compaction usage attribution unsupported; compaction not recorded");
    } else if (!entry || !entry.usage || typeof entry.usage !== "object") {
      warn("compaction usage missing; compaction not recorded");
    } else if (!compactionModel) {
      warn("compaction catalog model missing; compaction not recorded");
    } else {
      saveRecord(
        config,
        baseRecord(config, ctx, {
          timestamp: entry.timestamp,
          provider: compactionModel.provider,
          model: compactionModel.model,
          requestedModel: compactionModel.model,
          usage: entry.usage,
          eventKind: "compaction",
          attribution: "compaction_catalog_model",
        }),
      );
    }
    compactionModel = null;
    await flush(pi, config);
  });

  pi.on("agent_end", async () => {
    await flush(pi, config);
  });

  pi.on("session_shutdown", async () => {
    await flush(pi, config);
  });
}
