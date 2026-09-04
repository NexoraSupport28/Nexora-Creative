require("dotenv").config();
const {
  Client,
  Events,
  GatewayIntentBits,
  SlashCommandBuilder,
  REST,
  Routes,
  AttachmentBuilder,
  EmbedBuilder,
  ActivityType,
  ActionRowBuilder,
  ButtonBuilder,
  ButtonStyle,
  StringSelectMenuBuilder,
  ChannelType,
  PermissionFlagsBits,
  Status,
} = require("discord.js");
const axios = require("axios");
const os = require("os");
const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const { removeBackground } = require("@imgly/background-removal-node");

const MAX_FILE_SIZE = 8 * 1024 * 1024; // 8MB
const PROCESS_TIMEOUT_MS = 120000; // 2 minutes

// Images above this many pixels on the longest side skip the aspect-correction
// padding step, so very large images don't blow up memory on the square canvas.
const MAX_PAD_CANVAS = 4096;

// ---- Nexora AI (chat module) ---------------------------------------------
// When a channel is registered with /set, every non-command message sent there
// is answered by Google Gemini. The active channel per server is persisted to
// aiChannels.json so it survives bot restarts.
const DEFAULT_GEMINI_MODEL = "gemini-3.6-flash";
const DEFAULT_FALLBACK_MODELS = ["gemini-3.6-flash-lite", "gemini-3.6-pro"];
// Multi-model fallback chain: Nexora AI tries each model in order until one
// answers, so a retired, blocked, or rate-limited model never takes the chat
// down. The primary is always gemini-3.6-flash unless overridden.
//   GEMINI_MODEL  -> primary model (single, optional)
//   GEMINI_MODELS -> full chain as a comma-separated list (optional)
const GEMINI_MODELS = (() => {
  const primary = (process.env.GEMINI_MODEL || DEFAULT_GEMINI_MODEL).trim();
  const configured = (process.env.GEMINI_MODELS || "")
    .split(",")
    .map((m) => m.trim())
    .filter(Boolean);
  const chain = configured.length > 0 ? configured : [primary, ...DEFAULT_FALLBACK_MODELS];
  if (primary && !chain.includes(primary)) chain.unshift(primary);
  return chain;
})();
// Image generation uses Gemini's native image models ("Nano Banana" family).
// Each is tried in order until one produces an image:
//   GEMINI_IMAGE_MODEL  -> primary image model (single, optional)
//   GEMINI_IMAGE_MODELS -> full chain as a comma-separated list (optional)
const DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image";
const DEFAULT_IMAGE_FALLBACK_MODELS = ["gemini-3-pro-image", "gemini-2.5-flash-image"];
const IMAGE_MODELS = (() => {
  const primary = (process.env.GEMINI_IMAGE_MODEL || DEFAULT_IMAGE_MODEL).trim();
  const configured = (process.env.GEMINI_IMAGE_MODELS || "")
    .split(",")
    .map((m) => m.trim())
    .filter(Boolean);
  const chain = configured.length > 0 ? configured : [primary, ...DEFAULT_IMAGE_FALLBACK_MODELS];
  if (primary && !chain.includes(primary)) chain.unshift(primary);
  return chain;
})();
const NEXORA_SYSTEM_PROMPT =
  "You are Nexora AI, a friendly and helpful AI assistant in a Discord server. " +
  "You can see and analyze the files users attach to the conversation " +
  "(images, audio, video, PDFs, Word/Excel/PowerPoint, and source code). " +
  "Detect the language the user writes in — and when files are attached, also " +
  "consider the language of the file contents — then answer in that language " +
  "(Indonesian stays Indonesian, English stays English). If no language can be " +
  "clearly detected, answer in English. Keep replies concise and natural.";

const AI_CHANNELS_FILE = path.join(__dirname, "aiChannels.json");

function loadAiChannels() {
  try {
    return new Map(Object.entries(JSON.parse(fs.readFileSync(AI_CHANNELS_FILE, "utf8"))));
  } catch {
    return new Map(); // no file yet or unreadable -> start empty
  }
}

function saveAiChannels() {
  try {
    fs.writeFileSync(AI_CHANNELS_FILE, JSON.stringify(Object.fromEntries(aiChannels), null, 2));
  } catch (error) {
    console.error("⚠️ Failed to save Nexora AI channels:", error.message);
  }
}

const aiChannels = loadAiChannels();

// ---- Per-guild conversation memory (persisted to disk) --------------------
// Chat history is stored as one JSON file per server inside the memory/
// folder instead of RAM, so long conversations barely cost memory and they
// survive bot restarts. Capped per guild (see MAX_HISTORY_MESSAGES) and wiped
// by /newtask so long chats stay light.
const MEMORY_DIR = path.join(__dirname, "memory");

function memoryFilePath(guildId) {
  return path.join(MEMORY_DIR, `${guildId}.json`);
}

// Load the remembered conversation for a guild (empty array when none yet).
function loadConversation(guildId) {
  try {
    const parsed = JSON.parse(fs.readFileSync(memoryFilePath(guildId), "utf8"));
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return []; // no file yet, unreadable, or corrupt -> start fresh
  }
}

// Persist the conversation to disk (write to a temp file then rename, so a
// crash mid-write can never corrupt the stored history).
function saveConversation(guildId, history) {
  try {
    fs.mkdirSync(MEMORY_DIR, { recursive: true });
    const file = memoryFilePath(guildId);
    const tmp = `${file}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(history));
    fs.renameSync(tmp, file);
  } catch (error) {
    console.error(`⚠️ Failed to save conversation for guild ${guildId}:`, error.message);
  }
}

// Forget a guild's stored conversation. Returns true when a file was removed.
function deleteConversation(guildId) {
  try {
    const file = memoryFilePath(guildId);
    if (fs.existsSync(file)) {
      fs.unlinkSync(file);
      return true;
    }
  } catch (error) {
    console.error(`⚠️ Failed to delete conversation for guild ${guildId}:`, error.message);
  }
  return false;
}

// ---- Model health, rotation & rate-limit handling ---------------------------
// Google's API can rate-limit a model (HTTP 429) or drop it entirely (HTTP
// 404). Instead of failing the whole request the bot remembers each model's
// state:
//   • 429 -> model goes on a short cooldown honoring Google's "retry in Xs",
//            and the request is handed to the next healthy model.
//   • 404 -> model is marked unavailable and skipped for a while.
// The start of the chain rotates on every request, so load is spread across
// models instead of always hammering the first one.
const MODEL_RECHECK_AFTER_MS = 60 * 60 * 1000; // re-check a 404'd model after 1h
const MODEL_WAIT_CAP_MS = 120000; // never wait longer than this for a cooldown
const modelHealth = new Map(); // model -> { retryAt, unavailableUntil }
let modelRotation = 0;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Timestamp after which the model may be tried again (0 = healthy right now).
function modelBlockedUntil(model) {
  const health = modelHealth.get(model);
  if (!health) return 0;
  return Math.max(health.retryAt || 0, health.unavailableUntil || 0);
}

function markModelRateLimited(model, retrySeconds) {
  const retryMs = Math.max(2000, (retrySeconds || 30) * 1000);
  modelHealth.set(model, { ...(modelHealth.get(model) || {}), retryAt: Date.now() + retryMs });
  console.log(`⏳ Model "${model}" rate-limited — skipping it for ${Math.round(retryMs / 1000)}s.`);
}

function markModelUnavailable(model) {
  modelHealth.set(model, {
    ...(modelHealth.get(model) || {}),
    unavailableUntil: Date.now() + MODEL_RECHECK_AFTER_MS,
  });
  console.log(`🚫 Model "${model}" unavailable (404) — skipping it for ${MODEL_RECHECK_AFTER_MS / 60000} min.`);
}

// Models to try for this request: the chain rotated by one step (so every
// request starts on a different model) minus any that are cooling down.
function nextAttemptModels(models) {
  const offset = modelRotation % Math.max(models.length, 1);
  modelRotation += 1;
  const rotated = [...models.slice(offset), ...models.slice(0, offset)];
  return {
    rotated,
    ready: rotated.filter((model) => modelBlockedUntil(model) <= Date.now()),
  };
}

// Read Google's suggested wait from a 429 response ("Please retry in 45.9s").
function extractRetryDelaySeconds(error) {
  const searchable = [
    error?.response?.data ? JSON.stringify(error.response.data) : "",
    error?.response?.headers?.["retry-after"] ?? "",
    error?.message ?? "",
  ].join(" ");
  const match = searchable.match(/retry\s+in\s+([\d.]+)\s*s/i);
  if (match) return Math.ceil(parseFloat(match[1]));
  const header = Number(error?.response?.headers?.["retry-after"]);
  return Number.isFinite(header) && header > 0 ? Math.ceil(header) : undefined;
}

// Wait until the earliest cooling-down model is available again (bounded by
// MODEL_WAIT_CAP_MS so a stuck quota can never stall a request forever).
async function waitForRateLimitRecovery(models) {
  let earliest = Infinity;
  for (const model of models) {
    const blockedUntil = modelBlockedUntil(model);
    if (blockedUntil > Date.now() && blockedUntil < earliest) earliest = blockedUntil;
  }
  const waitMs = Math.min(earliest - Date.now(), MODEL_WAIT_CAP_MS);
  if (waitMs > 0) {
    console.log(`⏳ All models cooling down — retrying in ${Math.round(waitMs / 1000)}s.`);
    await sleep(waitMs);
  }
}

// ---- Explicit language detection (Indonesian vs English) --------------------
// Before asking Gemini we sample the user's message and any attached text/code
// files, then pick the dominant language from characteristic stop words. The
// result is passed to the model as an explicit instruction, so an Indonesian
// file does not get answered in English (or vice versa) — and an undetectable
// language simply falls back to the default (English).
const LANG_SAMPLE_CHARS = 3000;
const INDONESIAN_STOPWORDS = new Set([
  "yang", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "dan", "atau",
  "adalah", "tidak", "saya", "aku", "kamu", "kita", "kami", "mereka", "dia",
  "pada", "akan", "sudah", "bisa", "dapat", "juga", "karena", "tapi", "kalau",
  "jika", "apa", "bagaimana", "kenapa", "mengapa", "siapa", "kapan", "mana",
  "mau", "ingin", "punya", "ada", "dalam", "sebagai", "seperti", "oleh",
  "agar", "supaya", "sangat", "lebih", "kurang", "harus", "masih", "hanya",
  "semua", "setiap", "beberapa", "yaitu", "yakni", "antara", "terhadap",
  "tanpa", "setelah", "sebelum", "selama", "sejak", "ketika", "saat",
  "mungkin", "pernah", "sedang", "lagi", "jangan", "bilang", "kasih",
  "minta", "tolong", "wok", "bro", "gan", "kak", "bang", "nih", "deh",
  "dong", "yah", "ya", "gk", "ga", "nggak", "gak", "udah", "dah", "gimana",
  "kok", "loh", "lah", "sih", "tuh", "gitu", "gini", "aja", "doang", "biar",
  "bikin", "buat", "coba", "sama", "lain", "banyak", "sedikit", "bagus",
  "mantap", "keren", "gampang", "mudah", "susah", "sulit", "cepat", "lambat",
  "hari", "malam", "pagi", "siang", "sore", "kemarin", "besok", "sekarang",
  "tadi", "nanti", "disini", "disitu", "kesini", "kesitu", "disana", "kesana",
  "dulu", "kali", "rasanya", "kayaknya", "sepertinya", "cuma", "kok", "bgt",
]);
const ENGLISH_STOPWORDS = new Set([
  "the", "is", "are", "was", "were", "be", "been", "being", "and", "or", "of",
  "to", "in", "on", "for", "with", "by", "from", "at", "as", "an", "a", "it",
  "this", "that", "these", "those", "i", "you", "we", "they", "he", "she",
  "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
  "should", "may", "might", "must", "shall", "not", "but", "because", "if",
  "then", "so", "such", "than", "too", "what", "how", "why", "when", "where",
  "who", "which", "whose", "please", "thanks", "thank", "yes", "no", "also",
  "very", "more", "less", "most", "only", "just", "about", "into", "over",
  "under", "between", "during", "after", "before", "while", "up", "down",
  "out", "off", "again", "once", "here", "there", "all", "any", "both",
  "each", "few", "other", "some", "same", "own", "am", "me", "my", "your",
  "our", "their", "his", "her", "them", "us", "would", "could", "really",
  "actually", "maybe", "okay", "ok", "sure", "yes", "please", "help",
]);

function tokenizeForLanguage(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 1);
}

// Returns "indonesian", "english", or null when the language is unclear.
function detectLanguageOfText(text) {
  const tokens = tokenizeForLanguage((text || "").slice(0, LANG_SAMPLE_CHARS * 4));
  if (tokens.length === 0) return null;
  let indonesian = 0;
  let english = 0;
  for (const token of tokens) {
    if (INDONESIAN_STOPWORDS.has(token)) indonesian += 1;
    else if (ENGLISH_STOPWORDS.has(token)) english += 1;
  }
  const total = tokens.length;
  const idRatio = indonesian / total;
  const enRatio = english / total;
  if (idRatio > 0.02 && indonesian >= 2 && indonesian > english) return "indonesian";
  if (enRatio > 0.03 && english >= 3 && english > indonesian) return "english";
  if (indonesian > english && indonesian >= 3) return "indonesian";
  if (english > indonesian && english >= 3) return "english";
  return null;
}

// Sample the user message plus any attached text/code files (their real
// content), then run the detector over the combined sample.
async function detectLanguageOfRequest(userText, attachments) {
  const samples = [(userText || "").trim()];
  for (const attachment of attachments || []) {
    const displayName = attachment.name || "file";
    const ext = path.extname(displayName).slice(1).toLowerCase();
    const mime = normalizeAttachmentMime(attachment, ext);
    const kind = classifyAttachmentFile(mime, ext, displayName.toLowerCase());
    if (kind !== "text") continue; // binary media/docs are read by the model itself
    try {
      const buffer = await downloadFile(attachment.url);
      samples.push(buffer.toString("utf8").slice(0, LANG_SAMPLE_CHARS));
    } catch (error) {
      console.warn("⚠️ Could not sample attachment for language detection:", error.message);
    }
  }
  const combined = samples.join(" ");
  if (!combined.trim()) return null;
  return detectLanguageOfText(combined);
}

// Default analysis prompt, phrased in the detected language.
function defaultAnalysisPrompt(detectedLanguage) {
  return detectedLanguage === "indonesian"
    ? "Tolong analisis file yang dilampirkan, lalu jawab dalam bahasa Indonesia."
    : "Please analyze the attached file(s).";
}

// Ask Google Gemini to generate a reply for the given user text, using the
// remembered conversation history (per guild) for context. Attachments are
// analyzed in the same request (see buildUserTurnParts). Models are tried in
// a rotating order until one answers; rate-limited (429) and unavailable
// (404) models are handled so the chat survives quota hits and retired models.
async function askGemini(guildId, text, options = {}) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is missing in .env");
  }

  const {
    attachments = [],
    includeHistory = true,
    maxOutputTokens = MAX_OUTPUT_TOKENS,
    timeoutMs = 60000,
  } = options;

  // Detect the dominant language (user message + attached text files) so the
  // model is told explicitly which language to reply in.
  const detectedLanguage = await detectLanguageOfRequest(text, attachments);
  const languageInstruction =
    detectedLanguage === "indonesian"
      ? "IMPORTANT: The user's message and/or attached files are in Indonesian — reply in Indonesian."
      : detectedLanguage === "english"
        ? "IMPORTANT: The user's message and/or attached files are in English — reply in English."
        : null;
  const systemInstructionParts = [{ text: NEXORA_SYSTEM_PROMPT }];
  if (languageInstruction) systemInstructionParts.push({ text: languageInstruction });

  // Build contents from the remembered history + the new message. History
  // always alternates user/model because both sides are stored in pairs and
  // only ever contains text (files travel with a single message and are not
  // remembered across turns, which keeps memory light).
  const contents = [];
  if (includeHistory) {
    const history = loadConversation(guildId);
    for (const entry of history) {
      contents.push({ role: entry.role, parts: [{ text: entry.text }] });
    }
  }
  // The user's current turn: instruction text plus Gemini parts for every
  // attached file (images/audio/video/documents inline, text/code as text).
  // When the user only attached files without writing anything, use a default
  // prompt phrased in the detected language.
  const effectiveText =
    (text || "").trim() ||
    (attachments.length > 0 ? defaultAnalysisPrompt(detectedLanguage) : (text || ""));
  const userParts = await buildUserTurnParts(effectiveText, attachments);
  contents.push({ role: "user", parts: userParts });

  // Analyzing audio/video/documents takes longer than plain text.
  const effectiveTimeout =
    attachments.length > 0 ? Math.max(timeoutMs, 180000) : timeoutMs;

  const failures = [];
  const deadline = Date.now() + MODEL_WAIT_CAP_MS;
  while (Date.now() < deadline) {
    const { ready } = nextAttemptModels(GEMINI_MODELS);
    // Try the healthy models (rotated); fall back to the whole chain when
    // everything is cooling down (blocked ones are skipped inside the loop).
    const queue = ready.length > 0 ? ready : GEMINI_MODELS;
    let attempted = false;
    for (const model of queue) {
      if (modelBlockedUntil(model) > Date.now()) continue;
      attempted = true;
      try {
        const response = await axios.post(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
          {
            systemInstruction: { parts: systemInstructionParts },
            contents,
            generationConfig: { maxOutputTokens, temperature: 0.7 },
          },
          { timeout: effectiveTimeout }
        );
        const reply = response.data?.candidates?.[0]?.content?.parts
          ?.map((part) => part.text ?? "")
          .join("")
          .trim();
        if (!reply) {
          throw new Error("Gemini returned an empty response");
        }
        return { reply, model };
      } catch (error) {
        const status = error.response?.status;
        const detail = error.response?.data?.error?.message || error.message;
        failures.push(`${model}: ${status ? `HTTP ${status}` : "network"} — ${detail}`);
        if (status === 429) {
          // Rate limited: cool this model down and let the next one try.
          markModelRateLimited(model, extractRetryDelaySeconds(error));
        } else if (status === 404) {
          // Model retired / not enabled: stop wasting requests on it.
          markModelUnavailable(model);
        } else {
          console.warn(`⚠️ Model "${model}" failed${status ? ` (HTTP ${status})` : ""}: ${detail}`);
        }
      }
    }
    if (!attempted) {
      // Everyone is cooling down — wait for the earliest recovery, then retry.
      await waitForRateLimitRecovery(GEMINI_MODELS);
    } else {
      const allCooling = queue.every((model) => modelBlockedUntil(model) > Date.now());
      if (allCooling) {
        await waitForRateLimitRecovery(GEMINI_MODELS);
      } else {
        // Healthy models failed for other reasons: brief pause, then retry.
        await sleep(1000);
      }
    }
  }

  throw new Error(`All Gemini models failed. ${failures.join(" | ")}`);
}

// ---- Multimodal file analysis ---------------------------------------------
// Nexora AI understands attachments by sending them to Gemini in the same
// request as the user's text:
//   • images / audio / video / PDF / Word / Excel / PowerPoint  -> inline data
//   • plain text & source code                                  -> sent as text
// Gemini accepts far more inline data than Discord allows users to upload, so
// attachment.size is the real bottleneck (Discord caps ~25MB per file).
const MAX_ANALYSIS_FILE_SIZE = 24 * 1024 * 1024; // per file
const MAX_ANALYSIS_TOTAL_BYTES = 60 * 1024 * 1024; // combined per request
// Source files longer than this are truncated (token limits); a note tells the
// model the file was cut off.
const MAX_TEXT_ATTACHMENT_CHARS = 200000;

// Extension -> MIME type used when Discord reports application/octet-stream or
// no content type at all (common for code and some media files).
const EXTENSION_MIME = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
  gif: "image/gif",
  heic: "image/heic",
  heif: "image/heif",
  mp3: "audio/mpeg",
  wav: "audio/wav",
  ogg: "audio/ogg",
  oga: "audio/ogg",
  opus: "audio/ogg",
  flac: "audio/flac",
  aac: "audio/aac",
  m4a: "audio/mp4",
  aiff: "audio/aiff",
  aif: "audio/aiff",
  mp4: "video/mp4",
  m4v: "video/mp4",
  mov: "video/quicktime",
  webm: "video/webm",
  mkv: "video/x-matroska",
  avi: "video/x-msvideo",
  mpg: "video/mpeg",
  mpeg: "video/mpeg",
  wmv: "video/x-ms-wmv",
  flv: "video/x-flv",
  "3gp": "video/3gpp",
  "3gpp": "video/3gpp",
  pdf: "application/pdf",
  doc: "application/msword",
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  xls: "application/vnd.ms-excel",
  xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  ppt: "application/vnd.ms-powerpoint",
  pptx: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  rtf: "application/rtf",
};
// Every MIME in EXTENSION_MIME can be sent to Gemini as inline data.
const SUPPORTED_MEDIA_MIMES = new Set(Object.values(EXTENSION_MIME));

// Extensions treated as plain text / source code (read locally, then sent as a
// text part so Gemini never needs to know each language's MIME type).
const TEXT_FILE_EXTENSIONS = new Set([
  "txt", "md", "markdown", "log", "csv", "tsv", "json", "js", "mjs", "cjs",
  "jsx", "ts", "tsx", "py", "ipynb", "java", "c", "h", "cc", "cpp", "cxx",
  "hpp", "cs", "go", "rs", "php", "rb", "sh", "bash", "zsh", "ksh", "ps1",
  "bat", "cmd", "yaml", "yml", "xml", "html", "htm", "css", "scss", "sass",
  "less", "sql", "ini", "cfg", "conf", "toml", "env", "gitignore", "vue",
  "svelte", "kt", "kts", "swift", "scala", "lua", "pl", "pm", "r", "dart",
  "ex", "exs", "erl", "hrl", "clj", "cljs", "edn", "hs", "elm", "ml", "fs",
  "vb", "asm", "sol", "graphql", "gql", "proto", "tf", "tfvars", "hcl", "pug",
  "ejs", "hbs", "twig", "lock", "properties", "jsp", "asp", "aspx", "m", "mm",
  "gradle", "awk", "sed", "tex", "rtf",
]);
// Files whose *name* (no extension) marks them as text: Dockerfile, Makefile...
const TEXT_FILE_NAMES = new Set([
  "dockerfile", "makefile", "gemfile", "rakefile", "procfile", "license",
  "readme", "contributing", "changelog", "cmakelists.txt",
]);
// MIME types that are really text/code even though they don't start with text/.
const TEXT_MIME_TYPES = new Set([
  "application/json", "application/xml", "application/javascript",
  "application/x-javascript", "application/x-python", "application/x-sh",
  "application/x-csh", "application/x-httpd-php", "application/ld+json",
  "application/graphql",
]);

function analysisError(kind, message) {
  const error = new Error(message);
  error.isAnalysis = true;
  error.kind = kind;
  return error;
}

// Best-guess MIME for an attachment: trust Discord's content type unless it is
// missing or a generic application/octet-stream, then fall back to the
// extension mapping.
function normalizeAttachmentMime(attachment, ext) {
  const reported = (attachment.contentType || "").split(";")[0].trim().toLowerCase();
  if (reported && reported !== "application/octet-stream") return reported;
  return EXTENSION_MIME[ext] || "application/octet-stream";
}

// text -> read the bytes as UTF-8 and send them as a text part
// media -> send as inlineData (image/audio/video/document)
// unsupported -> the user gets a friendly error
function classifyAttachmentFile(mime, ext, fileName) {
  if (
    TEXT_FILE_EXTENSIONS.has(ext) ||
    TEXT_FILE_NAMES.has(fileName) ||
    mime.startsWith("text/") ||
    TEXT_MIME_TYPES.has(mime)
  ) {
    return "text";
  }
  if (SUPPORTED_MEDIA_MIMES.has(mime)) return "media";
  return "unsupported";
}

// Turn a message's attachments into Gemini content parts. Throws an
// analysisError with a user-friendly message when a file is too big or of an
// unsupported type.
async function prepareAttachmentParts(attachments) {
  const parts = [];
  let totalBytes = 0;
  for (let i = 0; i < attachments.length; i++) {
    const attachment = attachments[i];
    const displayName = attachment.name || `file_${i + 1}`;
    const fileName = displayName.toLowerCase();
    const ext = path.extname(displayName).slice(1).toLowerCase();

    if (attachment.size > MAX_ANALYSIS_FILE_SIZE) {
      throw analysisError(
        "too_large",
        `❌ **${displayName}** is ${formatBytes(attachment.size)} — the per-file limit for analysis is ${formatBytes(MAX_ANALYSIS_FILE_SIZE)}. Please send a smaller file.`
      );
    }
    totalBytes += attachment.size;
    if (totalBytes > MAX_ANALYSIS_TOTAL_BYTES) {
      throw analysisError(
        "too_large",
        `❌ These attachments total ${formatBytes(totalBytes)} — the combined limit is ${formatBytes(MAX_ANALYSIS_TOTAL_BYTES)} per message. Please send fewer or smaller files.`
      );
    }

    const mime = normalizeAttachmentMime(attachment, ext);
    const kind = classifyAttachmentFile(mime, ext, fileName);
    if (kind === "unsupported") {
      throw analysisError(
        "unsupported",
        `❌ **${displayName}** is a \`${mime}\` file, which I can't analyze. Supported: images (PNG/JPEG/WebP), audio (MP3/WAV/OGG/AAC/FLAC), video (MP4/MOV/WebM), PDF, Word, Excel, PowerPoint, and text/code files.`
      );
    }

    const buffer = await downloadFile(attachment.url);
    if (buffer.byteLength > MAX_ANALYSIS_FILE_SIZE) {
      throw analysisError(
        "too_large",
        `❌ **${displayName}** is ${formatBytes(buffer.byteLength)} after download — the per-file limit for analysis is ${formatBytes(MAX_ANALYSIS_FILE_SIZE)}.`
      );
    }

    if (kind === "text") {
      let content = buffer.toString("utf8");
      let truncated = false;
      if (content.length > MAX_TEXT_ATTACHMENT_CHARS) {
        content = content.slice(0, MAX_TEXT_ATTACHMENT_CHARS);
        truncated = true;
      }
      parts.push({
        text:
          `\n[Attachment ${i + 1}/${attachments.length}: \`${displayName}\`]\n` +
          "```\n" +
          content +
          "\n```" +
          (truncated
            ? `\n…(file was truncated after ${MAX_TEXT_ATTACHMENT_CHARS} characters)\n`
            : ""),
      });
    } else {
      // Label part first so Gemini knows which file each piece of media is.
      parts.push({
        text: `\n[Attachment ${i + 1}/${attachments.length}: \`${displayName}\` (${mime})]`,
      });
      parts.push({ inlineData: { mimeType: mime, data: buffer.toString("base64") } });
    }
  }
  return parts;
}

// Prepare images attached as *references* for image generation/editing
// (e.g. /imagine with a reference image). Only image MIME types are accepted;
// each image is downloaded and turned into a Gemini inlineData part.
async function prepareImageReferenceParts(attachments) {
  const parts = [];
  let totalBytes = 0;
  for (let i = 0; i < attachments.length; i++) {
    const attachment = attachments[i];
    const displayName = attachment.name || `image_${i + 1}`;
    const ext = path.extname(displayName).slice(1).toLowerCase();

    if (attachment.size > MAX_ANALYSIS_FILE_SIZE) {
      throw analysisError(
        "too_large",
        `❌ **${displayName}** is ${formatBytes(attachment.size)} — the per-file limit is ${formatBytes(MAX_ANALYSIS_FILE_SIZE)}. Please send a smaller image.`
      );
    }
    totalBytes += attachment.size;
    if (totalBytes > MAX_ANALYSIS_TOTAL_BYTES) {
      throw analysisError(
        "too_large",
        `❌ These images total ${formatBytes(totalBytes)} — the combined limit is ${formatBytes(MAX_ANALYSIS_TOTAL_BYTES)} per request.`
      );
    }

    const mime = normalizeAttachmentMime(attachment, ext);
    if (!(mime.startsWith("image/") && SUPPORTED_MEDIA_MIMES.has(mime))) {
      throw analysisError(
        "unsupported",
        `❌ **${displayName}** is a \`${mime}\` file — the reference must be an image (PNG, JPEG, WebP, HEIC, or GIF).`
      );
    }

    const buffer = await downloadFile(attachment.url);
    if (buffer.byteLength > MAX_ANALYSIS_FILE_SIZE) {
      throw analysisError(
        "too_large",
        `❌ **${displayName}** is ${formatBytes(buffer.byteLength)} after download — the per-file limit is ${formatBytes(MAX_ANALYSIS_FILE_SIZE)}.`
      );
    }

    parts.push({ text: `\n[Reference image ${i + 1}/${attachments.length}: \`${displayName}\`]` });
    parts.push({ inlineData: { mimeType: mime, data: buffer.toString("base64") } });
  }
  return parts;
}

// Build the "user" turn for Gemini: instruction text first, then Gemini
// content parts for every attached file.
async function buildUserTurnParts(text, attachments) {
  const parts = [];
  const prompt = (text || "").trim();
  if (!attachments || attachments.length === 0) {
    return [{ text: prompt || "..." }];
  }
  if (prompt) parts.push({ text: prompt });
  const fileParts = await prepareAttachmentParts(attachments);
  for (const part of fileParts) parts.push(part);
  if (!prompt) parts.unshift({ text: "Please analyze the attached file(s)." });
  return parts;
}

// Split a long reply into Discord-sized chunks (Discord caps messages at 2000
// characters), preferring to break on newlines / spaces.
function splitLongText(text, maxLength = 1900) {
  const message = String(text ?? "");
  if (message.length <= maxLength) return [message];
  const chunks = [];
  let rest = message;
  while (rest.length > maxLength) {
    let cut = rest.lastIndexOf("\n", maxLength);
    if (cut < maxLength / 2) cut = rest.lastIndexOf(" ", maxLength);
    if (cut < maxLength / 2) cut = maxLength;
    chunks.push(rest.slice(0, cut));
    rest = rest.slice(cut).trimStart();
  }
  if (rest) chunks.push(rest);
  return chunks;
}

// ---- Image generation -----------------------------------------------------
// Native Gemini image models ("Nano Banana") return the picture as base64 in
// candidates[].content.parts[].inlineData, requested with responseModalities.
const IMAGE_GENERATION_TIMEOUT_MS = 180000;

// referenceParts: optional inlineData parts for reference images. When present
// the prompt is treated as an *editing* instruction (text-to-image-to-image)
// instead of a from-scratch generation.
async function generateImageFromPrompt(prompt, referenceParts = []) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    const error = new Error("GEMINI_API_KEY is missing in .env");
    error.isImage = true;
    throw error;
  }

  // Instruction first, then the reference images it refers to.
  const parts = [{ text: prompt }, ...referenceParts];

  const failures = [];
  const deadline = Date.now() + MODEL_WAIT_CAP_MS;
  while (Date.now() < deadline) {
    const { ready } = nextAttemptModels(IMAGE_MODELS);
    const queue = ready.length > 0 ? ready : IMAGE_MODELS;
    let attempted = false;
    for (const model of queue) {
      if (modelBlockedUntil(model) > Date.now()) continue;
      attempted = true;
      try {
        const response = await axios.post(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
          {
            contents: [{ role: "user", parts }],
            generationConfig: { responseModalities: ["TEXT", "IMAGE"] },
          },
          { timeout: IMAGE_GENERATION_TIMEOUT_MS }
        );
        // Safety filters return no candidate but report promptFeedback.
        const blocked = response.data?.promptFeedback?.blockReason;
        if (blocked) throw new Error(`Request blocked by safety filters (${blocked})`);
        const contentParts = response.data?.candidates?.[0]?.content?.parts || [];
        for (const part of contentParts) {
          const mime = part.inlineData?.mimeType;
          const data = part.inlineData?.data;
          if (mime && data && mime.startsWith("image/")) {
            return { buffer: Buffer.from(data, "base64"), mime, model };
          }
        }
        throw new Error("Model returned no image");
      } catch (error) {
        const status = error.response?.status;
        const detail = error.response?.data?.error?.message || error.message;
        failures.push(`${model}: ${status ? `HTTP ${status}` : "network"} — ${detail}`);
        if (status === 429) {
          markModelRateLimited(model, extractRetryDelaySeconds(error));
        } else if (status === 404) {
          markModelUnavailable(model);
        } else {
          console.warn(`🎨 Image model "${model}" failed${status ? ` (HTTP ${status})` : ""}: ${detail}`);
        }
      }
    }
    if (!attempted) {
      await waitForRateLimitRecovery(IMAGE_MODELS);
    } else {
      const allCooling = queue.every((model) => modelBlockedUntil(model) > Date.now());
      if (allCooling) {
        await waitForRateLimitRecovery(IMAGE_MODELS);
      } else {
        await sleep(1000);
      }
    }
  }

  const error = new Error(`All image models failed. ${failures.join(" | ")}`);
  error.isImage = true;
  throw error;
}

// Wrap a generated image as a Discord attachment, recompressing to WebP when
// the raw output would exceed Discord's 8MB upload limit.
async function attachmentFromImageBuffer(buffer, mime, stem) {
  let output = buffer;
  let outputMime = mime && mime.startsWith("image/") ? mime : "image/png";
  let ext = outputMime === "image/jpeg" ? "jpg" : outputMime === "image/webp" ? "webp" : "png";
  if (output.byteLength > MAX_FILE_SIZE) {
    console.log(`📦 Generated image is ${formatBytes(output.byteLength)} — compressing to fit Discord's limit...`);
    output = await sharp(output).webp({ quality: 88 }).toBuffer();
    outputMime = "image/webp";
    ext = "webp";
    if (output.byteLength > MAX_FILE_SIZE) {
      throw new Error("Generated image is still too large for Discord even after compression");
    }
  }
  return new AttachmentBuilder(output, { name: `${stem}.${ext}` });
}

// Phrases that make Nexora route a chat message to the image generator
// instead of the text chat. /imagine is the explicit, always-reliable way.
const IMAGE_REQUEST_PATTERNS = [
  /^\/imagine\b/i,
  /^imagine\b/i,
  /^(?:buat(?:kan)?|bikin(?:in)?)\s+(?:aku|saya|kita|kami)?\s*(?:sebuah|suatu|satu)?\s*(?:gambar|foto|ilustrasi|logo|poster|kartun|meme)\b/i,
  /^gambar(?:kan|in)?\s+(?:aku|saya|untuk)\b/i,
  /^gambarkan\b/i,
  /^(?:tolong|bisa|minta|please)\s+(?:buat(?:kan)?|bikin|gambarkan)\s+(?:gambar|foto|ilustrasi|logo|poster)\b/i,
  /^(?:can you|could you|please)\s+(?:make|create|draw|generate)\b/i,
  /^(?:create|draw|generate|make|produce)\s+(?:me\s+)?(?:an?\s+|a\s+)?(?:image|picture|photo|drawing|artwork?|illustration|logo|poster)\b/i,
];

function looksLikeImageRequest(text) {
  const trimmed = (text || "").trim();
  return IMAGE_REQUEST_PATTERNS.some((pattern) => pattern.test(trimmed));
}

// Edit-style instructions ("ubah foto ini jadi anime", "remove the background",
// "ganti warna bajunya jadi merah", ...). Only treated as image edits when the
// message actually has an attached image — see the chat handler.
const IMAGE_EDIT_PATTERNS = [
  /^(?:ubah|edit|ganti|gantikan|tambah(?:kan)?|hapus|hilangkan|jadikan?|buat jadi|bikinin?|tolong (?:ubah|edit|ganti|tambah(?:kan)?|hapus|hilangkan))/i,
];

function looksLikeImageEditRequest(text) {
  const trimmed = (text || "").trim();
  return IMAGE_EDIT_PATTERNS.some((pattern) => pattern.test(trimmed));
}

// Bare imperative "gambar kucing" / "gambarin naga" (no verb "buatkan") means
// "draw/create". Guarded so questions about an existing image ("gambar ini
// apa?", "gambar kucing itu lucu ya") are never hijacked.
function looksLikeBareImageRequest(text) {
  const t = (text || "").trim();
  if (!/^(?:gambar|gambarin)\s/i.test(t)) return false;
  if (
    t.length > 120 ||
    /\?/.test(t) ||
    /\b(?:ini|itu|apa|yang|yg|mana|tersebut|tsb|dia|nih)\b/.test(t)
  ) {
    return false;
  }
  return true;
}

function imageGenerationErrorMessage(error) {
  const raw = (error && error.message) || String(error);
  const hint =
    "This usually means the configured image model can't produce images or your API key can't access it. " +
    "Set **GEMINI_IMAGE_MODEL** in `.env` to a current image model (e.g. `gemini-3.1-flash-image`), " +
    "restart the bot, and try again.";
  if (raw.includes("GEMINI_API_KEY")) return `❌ **GEMINI_API_KEY** is missing in \`.env\`.`;
  if (raw.includes("blocked by safety filters")) return "❌ Image generation was blocked by safety filters — try a different prompt.";
  if (raw.includes("All image models failed")) return `❌ Image generation failed. ${hint}`;
  return `❌ Image generation failed: ${raw} ${hint}`;
}

// ---- /imagine & /analyze handlers ----------------------------------------
async function handleImagineCommand(interaction) {
  const prompt = (interaction.options.getString("prompt") || "").trim();
  if (!process.env.GEMINI_API_KEY) {
    await interaction.reply({
      content:
        "❌ **GEMINI_API_KEY** is not set in `.env`. Add your Google Gemini API key, " +
        "restart the bot, then run `/imagine` again.",
      ephemeral: true,
    });
    return;
  }
  if (!prompt) {
    await interaction.reply({
      content: "❌ Please describe the image you want to create.",
      ephemeral: true,
    });
    return;
  }

  await interaction.deferReply();
  try {
    // Optional reference image -> the request becomes an image *edit*.
    const reference = interaction.options.getAttachment("reference");
    const referenceParts = reference ? await prepareImageReferenceParts([reference]) : [];
    const { buffer, mime, model } = await generateImageFromPrompt(prompt, referenceParts);
    const file = await attachmentFromImageBuffer(buffer, mime, "generated_image");
    const caption = referenceParts.length > 0 ? "edited your image" : "your image";
    await interaction.editReply({
      content: `🎨 Here's ${caption}!\n**${prompt}**`,
      files: [file],
    });
    if (model !== IMAGE_MODELS[0]) console.log(`🎨 Generated image with fallback model: ${model}`);
  } catch (error) {
    console.error("❌ /imagine error:", error.message);
    const message = error.message || String(error);
    await interaction
      .editReply({
        content: error.isAnalysis || message.startsWith("❌") ? message : imageGenerationErrorMessage(error),
      })
      .catch(() => {});
  }
}

async function handleAnalyzeCommand(interaction) {
  const attachment = interaction.options.getAttachment("file");
  if (!process.env.GEMINI_API_KEY) {
    await interaction.reply({
      content:
        "❌ **GEMINI_API_KEY** is not set in `.env`. Add your Google Gemini API key, " +
        "restart the bot, then run `/analyze` again.",
      ephemeral: true,
    });
    return;
  }
  if (!attachment) {
    await interaction.reply({
      content: "❌ Please attach a file to analyze.",
      ephemeral: true,
    });
    return;
  }

  await interaction.deferReply();
  try {
    // One-shot analysis: no conversation memory, a longer answer budget.
    const { reply, model } = await askGemini(interaction.guildId, "", {
      attachments: [attachment],
      includeHistory: false,
      maxOutputTokens: MAX_OUTPUT_TOKENS * 2,
    });
    if (model !== GEMINI_MODELS[0]) console.log(`🤖 /analyze answered with fallback model: ${model}`);
    const chunks = splitLongText(`📄 **Analysis of ${attachment.name}:**\n\n${reply}`);
    await interaction.editReply(chunks[0]);
    for (const chunk of chunks.slice(1)) {
      await interaction.followUp(chunk);
    }
  } catch (error) {
    console.error("❌ /analyze error:", error.message);
    const message = error.message || String(error);
    await interaction
      .editReply({
        content: error.isAnalysis || message.startsWith("❌") ? message : `❌ Failed to analyze the file. ${message}`,
      })
      .catch(() => {});
  }
}

// ---- Queue configuration -------------------------------------------------
// At most MAX_CONCURRENT_TASKS images are processed at the same time.
// Everyone else waits in a FIFO queue and gets a live status panel.
const envInt = (name, fallback, min = 0) => {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  return Number.isFinite(value) && value >= min ? Math.floor(value) : fallback;
};
const MAX_CONCURRENT_TASKS = envInt("MAX_CONCURRENT_TASKS", 20, 1);
const MAX_WAITING_JOBS = envInt("MAX_WAITING_JOBS", 60, 1);
const QUEUE_UPDATE_INTERVAL_MS = envInt("QUEUE_UPDATE_INTERVAL_MS", 4000, 0);
// Nexora AI chat settings: reply length cap and per-guild memory size (RAM).
const MAX_OUTPUT_TOKENS = envInt("MAX_OUTPUT_TOKENS", 1024, 64);
const MAX_HISTORY_MESSAGES = envInt("MAX_HISTORY_MESSAGES", 20, 1);
// How often the background probe re-measures latency. The bot presence and
// /status read these samples, so the value shown is never more than a few
// seconds old (the gateway heartbeat alone would be up to ~40s stale).
const LATENCY_SAMPLE_INTERVAL_MS = envInt("LATENCY_SAMPLE_INTERVAL_MS", 5000, 1000);
const LATENCY_HISTORY_MAX = 12; // rolling window (~1 minute at 5s per sample)

// Fail fast with a clear message if the bot token is missing
if (!process.env.DISCORD_TOKEN) {
  console.error("❌ DISCORD_TOKEN is missing. Add it to your .env file.");
  process.exit(1);
}

// Shared REST client: used for command registration and the live /status
// latency probe (token presence is guaranteed by the check above).
const rest = new REST({ version: "10" }).setToken(process.env.DISCORD_TOKEN);

// Bot client setup
const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

// Slash command definition
const commands = [
  new SlashCommandBuilder()
    .setName("removebg")
    .setDescription("Remove background from an image")
    .addAttachmentOption((option) =>
      option
        .setName("image")
        .setDescription("The image to remove background from")
        .setRequired(true)
    ),
  new SlashCommandBuilder()
    .setName("status")
    .setDescription("Check bot status: latency, CPU, RAM, uptime, and more"),
  new SlashCommandBuilder()
    .setName("set")
    .setDescription("Set this channel's role: Nexora AI chat or Support Panel")
    .addStringOption((option) =>
      option
        .setName("module")
        .setDescription("The module to set up in this channel")
        .setRequired(true)
        .addChoices(
          { name: "Nexora AI", value: "nexora_ai" },
          { name: "Support Panel", value: "support_panel" },
          { name: "Off", value: "off" }
        )
    ),
  new SlashCommandBuilder()
    .setName("newtask")
    .setDescription("Start a fresh Nexora AI conversation (forgets chat history)"),
  new SlashCommandBuilder()
    .setName("analyze")
    .setDescription("Analyze an image, video, audio, PDF, Word, Excel, PowerPoint, or code file")
    .addAttachmentOption((option) =>
      option.setName("file").setDescription("The file to analyze").setRequired(true)
    ),
  new SlashCommandBuilder()
    .setName("imagine")
    .setDescription("Generate an image from a text description (attach a reference image to edit it)")
    .addStringOption((option) =>
      option
        .setName("prompt")
        .setDescription("Describe the image you want, e.g. a red dragon flying over a castle")
        .setRequired(true)
        .setMinLength(1)
        .setMaxLength(2000)
    )
    .addAttachmentOption((option) =>
      option
        .setName("reference")
        .setDescription("Optional image to use as a reference — describe the change you want in the prompt")
        .setRequired(false)
    ),
].map((command) => command.toJSON());

// Register slash commands
async function registerCommands() {
  try {
    console.log("🔄 Registering slash commands...");
    await rest.put(Routes.applicationCommands(client.user.id), {
      body: commands,
    });
    console.log("✅ Slash commands registered successfully!");
  } catch (error) {
    console.error("❌ Error registering commands:", error);
  }
}

// ---- Bot status / presence ----------------------------------------------
function formatBytes(bytes) {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${bytes} B`;
}

function formatUptime(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const days = Math.floor(totalSeconds / 86400);
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const parts = [];
  if (days) parts.push(`${days}d`);
  if (hours) parts.push(`${hours}h`);
  if (minutes) parts.push(`${minutes}m`);
  parts.push(`${seconds}s`);
  return parts.join(" ");
}

// Sample real CPU usage by measuring process CPU time over a short window.
async function cpuUsagePercent(sampleMs = 100) {
  const start = process.cpuUsage();
  await new Promise((resolve) => setTimeout(resolve, sampleMs));
  const delta = process.cpuUsage(start);
  const usedMs = (delta.user + delta.system) / 1000; // microseconds -> ms
  const cores = os.cpus().length;
  return Math.min((usedMs / sampleMs / cores) * 100, 100).toFixed(1);
}

// ---- Status UI -----------------------------------------------------------
// Colors used across the status panel; the overall health decides the accent.
const STATUS_COLORS = {
  good: 0x22c55e,
  warn: 0xf59e0b,
  bad: 0xef4444,
  info: 0x3b82f6,
  mute: 0x6b7280,
};

// A gateway heartbeat is only answered roughly every ~40s, so client.ws.ping
// starts at -1 (no heartbeat yet) and can go stale between heartbeats.
// Anything that isn't a positive number counts as "not measured yet".
function validPing(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function latencyTier(ms) {
  if (!validPing(ms)) {
    return { emoji: "❔", label: "Not measured yet", color: STATUS_COLORS.mute };
  }
  if (ms < 100) return { emoji: "🟢", label: "Excellent", color: STATUS_COLORS.good };
  if (ms < 200) return { emoji: "🟡", label: "Good", color: STATUS_COLORS.good };
  if (ms < 400) return { emoji: "🟠", label: "Okay", color: STATUS_COLORS.warn };
  return { emoji: "🔴", label: "Slow", color: STATUS_COLORS.bad };
}

// Emoji progress bar, colored by how loaded the resource is.
function usageBar(percent, length = 10) {
  const clamped = Math.max(0, Math.min(100, percent));
  const filled = Math.round((clamped / 100) * length);
  const cell = clamped >= 90 ? "🟥" : clamped >= 70 ? "🟧" : "🟩";
  return `${cell.repeat(filled)}${"⬜".repeat(length - filled)}`;
}

// Discord gateway connection state (this bot runs on a single shard).
function gatewayStateText() {
  const shard = client.ws.shards?.first?.();
  if (!shard) return "⏳ Connecting…";
  switch (shard.status) {
    case Status.Ready:
      return "✅ Connected";
    case Status.Resuming:
      return "📡 Resuming session…";
    case Status.Identifying:
      return "🔑 Identifying…";
    case Status.WaitingForGuilds:
      return "🚀 Catching up on guilds…";
    case Status.Reconnecting:
      return "🔄 Reconnecting…";
    case Status.Connecting:
      return "⏳ Connecting…";
    case Status.Disconnected:
      return "❌ Disconnected";
    default:
      return "💤 Idle";
  }
}

// ---- Background latency sampler -------------------------------------------
// REST has no heartbeat, so instead of relying on the gateway's ~40s-old
// heartbeat ping this bot measures its own latency: one timed request to
// Discord every LATENCY_SAMPLE_INTERVAL_MS (5s by default). Presence text and
// /status both read the rolling snapshot, so every shown latency is at most a
// few seconds old and re-measured continuously.
const latencyStore = {
  history: [], // recent sample values (ms), newest at the end
  updatedAt: 0,
  sampling: false,
  failing: false,
};

// One timed REST round-trip. Never overlaps the previous sample and disables
// auto-retry so a dead network can't stall the loop.
async function sampleLatencyOnce() {
  if (latencyStore.sampling) return;
  latencyStore.sampling = true;
  try {
    const start = process.hrtime.bigint();
    await rest.get(Routes.user(), { retries: 0 });
    const ms = Math.round(Number(process.hrtime.bigint() - start) / 1e6); // ns -> ms
    latencyStore.history.push(ms);
    if (latencyStore.history.length > LATENCY_HISTORY_MAX) latencyStore.history.shift();
    latencyStore.updatedAt = Date.now();
    if (latencyStore.failing) {
      latencyStore.failing = false;
      console.log("✅ Latency sampling recovered.");
    }
  } catch (error) {
    // Log once when it starts failing, not on every 5s tick.
    if (!latencyStore.failing) {
      latencyStore.failing = true;
      console.error("⚠️ Latency sampling failing (keeps retrying every tick):", error.message);
    }
  } finally {
    latencyStore.sampling = false;
  }
}

// Rolling snapshot: latest sample plus best/average over the window.
function latencySnapshot() {
  const { history, updatedAt } = latencyStore;
  if (history.length === 0) return null;
  return {
    latest: history[history.length - 1],
    best: Math.min(...history),
    avg: Math.round(history.reduce((a, b) => a + b, 0) / history.length),
    samples: history.length,
    ageSec: Math.max(0, Math.round((Date.now() - updatedAt) / 1000)),
  };
}

// The gateway's heartbeat ping plus how many seconds ago it was sampled.
// A heartbeat ack only arrives roughly every ~40s, so a high age means the
// value is stale and the live API probe should be trusted instead.
function gatewayPingInfo() {
  const shard = client.ws.shards?.first?.();
  if (!shard || !validPing(shard.ping)) return null;
  const lastAt = typeof shard.lastPingTimestamp === "number" ? shard.lastPingTimestamp : -1;
  const ageSec = lastAt > 0 ? Math.max(0, Math.round((Date.now() - lastAt) / 1000)) : null;
  return { ping: shard.ping, ageSec };
}

function liveLatencyFieldText(snapshot) {
  if (!snapshot) return "❔ **—**\nFirst sample pending — hit 🔄 Refresh";
  const tier = latencyTier(snapshot.best);
  return `${tier.emoji} **${snapshot.latest} ms**\nbest of ${snapshot.samples} · avg ${snapshot.avg} ms · ${snapshot.ageSec}s ago`;
}

function gatewayLatencyFieldText(gateway) {
  if (!gateway) return "❔ **—**\nWaiting for first heartbeat…";
  const tier = latencyTier(gateway.ping);
  const freshness =
    gateway.ageSec === null
      ? "age unknown"
      : gateway.ageSec <= 5
        ? "sampled just now"
        : gateway.ageSec <= 20
          ? `sampled ${gateway.ageSec}s ago`
          : `sampled ${gateway.ageSec}s ago (stale)`;
  return `${tier.emoji} **${Math.round(gateway.ping)} ms**\n${freshness}`;
}

// Pick the worst tier among the live metrics -> accent color + one-line summary.
function overallHealth(cpu, memPercent, latency) {
  const live = validPing(latency) ? latency : null;
  if (cpu >= 90 || memPercent >= 90 || (live !== null && live >= 400)) {
    return { emoji: "🔴", color: STATUS_COLORS.bad, text: "Under heavy load — expect slower responses." };
  }
  if (cpu >= 70 || memPercent >= 70 || (live !== null && live >= 200)) {
    return { emoji: "🟠", color: STATUS_COLORS.warn, text: "A bit busy, but everything is still running fine." };
  }
  if (live === null) {
    return { emoji: "🔵", color: STATUS_COLORS.info, text: "Connected — first latency sample still coming in." };
  }
  return { emoji: "🟢", color: STATUS_COLORS.good, text: "All systems operational — everything is running smoothly." };
}

// The Refresh button attached to the /status panel.
function statusActionRow() {
  return new ActionRowBuilder().addComponents(
    new ButtonBuilder()
      .setCustomId("status_refresh")
      .setLabel("Refresh")
      .setEmoji("🔄")
      .setStyle(ButtonStyle.Secondary)
  );
}

async function statusEmbed() {
  const totalMem = os.totalmem();
  const usedMem = totalMem - os.freemem();
  const memPercent = Math.round((usedMem / totalMem) * 100);
  const serverCount = client.guilds.cache.size;
  const userCount = client.guilds.cache.reduce(
    (sum, guild) => sum + (guild.memberCount || 0),
    0
  );
  const channelCount = client.channels.cache.size;

  const cpu = Number(await cpuUsagePercent());
  const gateway = gatewayPingInfo(); // ws heartbeat sample + how old it is
  const snapshot = latencySnapshot(); // rolling 5s background samples
  // Health uses the smoothed average once enough samples exist, otherwise the
  // latest sample; the (possibly stale) gateway heartbeat is only a fallback
  // until the very first background sample lands.
  const healthLatency = snapshot
    ? snapshot.samples >= 3
      ? snapshot.avg
      : snapshot.latest
    : gateway?.ping ?? null;
  const health = overallHealth(cpu, memPercent, healthLatency);

  return new EmbedBuilder()
    .setColor(health.color)
    .setTitle("📊 Bot Status")
    .setDescription(
      `${health.emoji} **${health.text}**\nCurrent status of **${client.user.username}**.`
    )
    .addFields(
      { name: "⚡ Live ping", value: liveLatencyFieldText(snapshot), inline: true },
      { name: "🌐 Gateway (WS)", value: gatewayLatencyFieldText(gateway), inline: true },
      { name: "📡 Connection", value: gatewayStateText(), inline: true },
      {
        name: "💻 CPU Usage",
        value: `${usageBar(cpu)}\n**${cpu.toFixed(1)}%** of ${os.cpus().length} cores`,
        inline: true,
      },
      {
        name: "🧠 System RAM",
        value: `${usageBar(memPercent)}\n**${formatBytes(usedMem)}** / ${formatBytes(totalMem)} (**${memPercent}%**)`,
        inline: true,
      },
      {
        name: "📦 Process RAM",
        value: `RSS **${formatBytes(process.memoryUsage().rss)}**\nHeap **${formatBytes(
          process.memoryUsage().heapUsed
        )}**`,
        inline: true,
      },
      {
        name: "🖥️ Host",
        value: `${os.platform()} ${os.arch()}\nNode ${process.version}`,
        inline: true,
      },
      {
        name: "🏗️ discord.js",
        value: `v${require("discord.js").version}\nHost up: ${formatUptime(os.uptime() * 1000)}`,
        inline: true,
      },
      { name: "🤖 Bot uptime", value: formatUptime(client.uptime), inline: true },
      {
        name: "📈 Stats",
        value: `**${serverCount}** servers\n**${userCount}** users\n**${channelCount}** channels`,
        inline: true,
      },
      {
        name: "⏳ Queue",
        value: `**${activeCount}** / ${MAX_CONCURRENT_TASKS} active\n**${waitingQueue.length}** waiting (max ${MAX_WAITING_JOBS})`,
        inline: true,
      }
    )
    .setFooter({ text: "Click 🔄 Refresh for live numbers" })
    .setTimestamp();
}

// Live "watching" presence: server count + the freshest sampled latency, and it
// runs right after every latency sample (default: every 5 seconds).
// NOTE: in discord.js v14 setActivity() is synchronous and returns a
// ClientPresence — NOT a Promise — so it must never have .catch() chained on
// it (that throws "setActivity(...).catch is not a function").
let lastPresenceText = "";

function updatePresence() {
  if (!client.user) return;
  const serverCount = client.guilds.cache.size;
  const snapshot = latencySnapshot();
  const latency = snapshot ? `${snapshot.latest}ms` : "n/a";
  const text = `Creative Server: ${serverCount} | Latency: ${latency}`;
  if (text === lastPresenceText) return; // skip redundant gateway updates
  lastPresenceText = text;
  try {
    client.user.setActivity(text, { type: ActivityType.Watching });
  } catch (error) {
    console.error("⚠️ Failed to update presence:", error.message);
  }
}

// Download a file (image, video, audio, document, ...) with axios and retry logic
async function downloadFile(url, retries = 3) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      console.log(`📥 Downloading file (attempt ${attempt}/${retries})...`);
      const response = await axios.get(url, {
        responseType: "arraybuffer",
        timeout: 60000, // 60 second timeout
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
      });
      console.log("✅ File downloaded successfully!");
      return Buffer.from(response.data);
    } catch (error) {
      console.error(`❌ Download attempt ${attempt} failed:`, error.message);
      if (attempt === retries) {
        throw new Error(`Failed to download file after ${retries} attempts: ${error.message}`);
      }
      // Wait before retry
      await new Promise((resolve) => setTimeout(resolve, 2000 * attempt));
    }
  }
}

// Build a friendly error message for the user
function friendlyError(error) {
  const errMsg = error && error.message ? error.message : String(error);
  let errorMessage = "❌ Failed to process image. ";
  if (errMsg.includes("download")) {
    errorMessage += "Could not download the image. Please try again.";
  } else if (errMsg.includes("timed out")) {
    errorMessage += "Processing took too long. Try a smaller image.";
  } else if (errMsg.includes("too large")) {
    errorMessage += "Result is too large for Discord.";
  } else {
    errorMessage += "Please try with a different image.";
  }
  return errorMessage;
}

// ---- Task queue ----------------------------------------------------------
const waitingQueue = []; // FIFO of { interaction, position, lastUpdateAt, panelDead }
let activeCount = 0;

function queuePanelEmbed(entry) {
  return new EmbedBuilder()
    .setColor(0xf0a020)
    .setTitle("⏳ In queue — please wait")
    .setDescription(
      "All processing slots are busy right now. Your image will be processed automatically " +
        "as soon as a slot opens — you don't need to do anything."
    )
    .addFields(
      { name: "Position", value: `#${entry.position}`, inline: true },
      { name: "Ahead of you", value: `${Math.max(entry.position - 1, 0)}`, inline: true },
      { name: "Active jobs", value: `${activeCount}/${MAX_CONCURRENT_TASKS}`, inline: true }
    )
    .setFooter({ text: "This panel updates live." });
}

function processingPanelEmbed() {
  return new EmbedBuilder()
    .setColor(0x3b82f6)
    .setTitle("🔄 Processing your image…")
    .setDescription("This usually takes a few seconds. Hang tight!");
}

function donePanelEmbed() {
  return new EmbedBuilder()
    .setColor(0x22c55e)
    .setTitle("✅ Background removed!")
    .setDescription("Your image is ready — thanks for using the bot! 💙")
    .setFooter({ text: "Use /removebg to remove another background" });
}

async function editQueuePanel(entry) {
  if (entry.updating) return;
  entry.updating = true;
  try {
    await entry.interaction.editReply({ embeds: [queuePanelEmbed(entry)] });
    entry.lastUpdateAt = Date.now();
  } catch (error) {
    // Interaction may have expired while the user waited; stop touching it.
    console.error("⚠️ Failed to update queue panel:", error.message);
    entry.panelDead = true;
  } finally {
    entry.updating = false;
  }
}

// Update the position shown on every waiting user's panel (throttled so a
// long queue cannot spam the Discord rate limit).
async function updateWaitingPanels() {
  const jobs = [];
  for (let i = 0; i < waitingQueue.length; i++) {
    const entry = waitingQueue[i];
    const position = i + 1;
    const changed = position !== entry.position;
    entry.position = position;
    if (entry.panelDead || entry.updating || !changed) continue;
    if (Date.now() - entry.lastUpdateAt < QUEUE_UPDATE_INTERVAL_MS) continue;
    jobs.push(editQueuePanel(entry));
  }
  await Promise.allSettled(jobs);
}

// ---- Background removal core (shared by /removebg and chat requests) ------
// Runs the local background-removal model on an already-downloaded image and
// returns a finished PNG buffer no larger than Discord's 8MB upload limit.
async function removeBackgroundFromBuffer(imageBuffer, contentType) {
  // ---- Aspect-correct inference ----
  // The library resizes every image to a 1024x1024 square before running the
  // model. For non-square photos that stretches the subject, which makes the
  // cutout less precise. Padding the input to a square canvas first keeps the
  // subject undistorted during inference; we crop back to the exact original
  // pixel dimensions afterwards, so the result keeps its original resolution.
  const metadata = await sharp(imageBuffer).metadata();
  const { width, height } = metadata;
  let inferenceBlob = new Blob([imageBuffer], { type: contentType });
  let cropRect = null;
  if (width && height && width !== height && Math.max(width, height) <= MAX_PAD_CANVAS) {
    const side = Math.max(width, height);
    console.log(`📐 Padding ${width}x${height} to ${side}x${side} for accurate inference...`);
    const padded = await sharp({
      create: {
        width: side,
        height: side,
        channels: 4,
        background: { r: 0, g: 0, b: 0, alpha: 0 }, // transparent pad
      },
    })
      .composite([{ input: imageBuffer }]) // original image anchored top-left
      .png()
      .toBuffer();
    inferenceBlob = new Blob([padded], { type: "image/png" });
    cropRect = { left: 0, top: 0, width, height };
  }

  // Remove background with timeout.
  // The image is passed as a Blob carrying its MIME type (the library decodes
  // based on the blob's type; raw Buffers/absolute Windows paths do not work).
  console.log("🎨 Removing background...");
  let lastLoggedPercent = -1;
  const result = await Promise.race([
    removeBackground(new Blob([imageBuffer], { type: contentType }), {
      output: { format: "image/png" },
      model: "medium",
      progress: (key, current, total) => {
        const percent = total > 0 ? Math.floor((current / total) * 100) : 0;
        if (percent !== lastLoggedPercent && (percent === 0 || percent === 100 || percent - lastLoggedPercent >= 10)) {
          lastLoggedPercent = percent;
          console.log(`⏳ Progress: ${key} - ${percent}%`);
        }
      },
    }),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Background removal timed out")), PROCESS_TIMEOUT_MS)
    ),
  ]);

  // removeBackground resolves to a Blob, which fs/Buffer APIs can't consume
  // directly, so convert it to a Buffer.
  let resultBuffer = Buffer.from(await result.arrayBuffer());

  // Restore the original (possibly non-square) resolution by removing the pad.
  if (cropRect) {
    console.log("✂️ Restoring original resolution...");
    resultBuffer = await sharp(resultBuffer).extract(cropRect).png().toBuffer();
  }

  // Check result file size (Discord limit is 8MB for bots)
  if (resultBuffer.byteLength > MAX_FILE_SIZE) {
    throw new Error("Result image too large for Discord");
  }

  return resultBuffer;
}

// Image types accepted by the background remover.
const BG_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);

function isBgRemovableAttachment(attachment) {
  return (
    attachment &&
    attachment.contentType &&
    BG_IMAGE_TYPES.has(attachment.contentType) &&
    attachment.size <= MAX_FILE_SIZE
  );
}

// Phrases that mean "remove the background of this image" in a chat message.
// Only acted on when the message actually carries an image attachment.
function looksLikeRemoveBgRequest(text) {
  const t = (text || "").trim();
  return (
    /\bremove\s+(?:the\s+)?(?:background|bg)\b/i.test(t) ||
    /\bbg\s*remov(?:e|al|er|ing)?\b/i.test(t) ||
    /\bremovebg\b/i.test(t) ||
    /\b(?:hapus|buang|hilangkan|ilangin)\s+(?:background|bg|latar(?:\s+belakang)?)(?:nya)?\b/i.test(t) ||
    /\b(?:background|bg|latar(?:\s+belakang)?)(?:nya)?\s+(?:di)?hapus\b/i.test(t) ||
    /\b(?:background|latar(?:\s+belakang)?)\s*remover\b/i.test(t) ||
    /\b(?:buat|bikin|jadikan?|ubah)\s+(?:background|bg|latar(?:\s+belakang)?)?\s*transparan\b/i.test(t) ||
    /\b(?:background|bg|latar(?:\s+belakang)?)\s+transparan\b/i.test(t)
  );
}

// Chat-originated background removal: same pipeline as /removebg, replying to
// the message instead of editing a slash-command panel. Takes one slot from the
// shared pool so the local model never runs unbounded.
async function runChatBackgroundRemoval(message, attachment) {
  if (activeCount >= MAX_CONCURRENT_TASKS) {
    await message
      .reply(
        "⏳ All background-removal slots are busy right now — try again in a moment, or use `/removebg` which queues automatically."
      )
      .catch(() => {});
    return;
  }
  activeCount += 1;
  try {
    await message.channel.sendTyping();
    console.log("🧹 Chat remove-bg request detected — processing image...");
    const imageBuffer = await downloadFile(attachment.url);
    const resultBuffer = await removeBackgroundFromBuffer(imageBuffer, attachment.contentType);
    const resultAttachment = new AttachmentBuilder(resultBuffer, {
      name: "removed_background.png",
    });
    await message.reply({
      content: "✅ **Background removed!** 💙",
      files: [resultAttachment],
    });
    console.log("✅ Chat remove-bg processed and sent!");
  } catch (error) {
    console.error("❌ Chat remove-bg error:", error.message);
    await message.reply(friendlyError(error)).catch(() => {});
  } finally {
    activeCount -= 1;
    dispatch();
  }
}

// The actual /removebg processing. Everything from download to sending the result.
async function runJob({ interaction, attachment }) {
  try {
    await interaction.editReply({ embeds: [processingPanelEmbed()] });

    // Download the image with retry
    const imageBuffer = await downloadFile(attachment.url);

    // Remove the background (aspect-correction padding & cropping handled inside)
    const resultBuffer = await removeBackgroundFromBuffer(imageBuffer, attachment.contentType);

    // Send result: the loading panel is updated into a green "done" panel,
    // and the finished image is attached alongside it.
    const resultAttachment = new AttachmentBuilder(resultBuffer, {
      name: "removed_background.png",
    });

    await interaction.editReply({
      embeds: [donePanelEmbed()],
      files: [resultAttachment],
    });

    console.log("✅ Image processed and sent!");
  } catch (error) {
    console.error("❌ Error processing image:", error);
    try {
      if (interaction.deferred || interaction.replied) {
        await interaction.editReply({ content: friendlyError(error) });
      } else {
        await interaction.reply({ content: friendlyError(error), ephemeral: true });
      }
    } catch (replyError) {
      console.error("❌ Failed to send error reply:", replyError.message);
    }
  }
}

function dispatch() {
  while (activeCount < MAX_CONCURRENT_TASKS && waitingQueue.length > 0) {
    const entry = waitingQueue.shift();
    activeCount += 1;
    runJob(entry)
      .catch((error) => console.error("❌ Unexpected job error:", error))
      .finally(() => {
        activeCount -= 1;
        dispatch();
        updateWaitingPanels();
      });
  }
}

// Add a request to the queue. If a slot is free it starts immediately;
// otherwise the user receives a live queue panel.
async function enqueueRequest(interaction, attachment) {
  if (waitingQueue.length >= MAX_WAITING_JOBS) {
    await interaction.editReply({
      content:
        `❌ The queue is full (${MAX_CONCURRENT_TASKS} running + ${MAX_WAITING_JOBS} waiting). ` +
        "Please try again in a few minutes.",
    });
    return;
  }

  const entry = {
    interaction,
    attachment,
    position: waitingQueue.length + 1,
    lastUpdateAt: 0,
    panelDead: false,
  };
  waitingQueue.push(entry);

  if (activeCount < MAX_CONCURRENT_TASKS) {
    dispatch(); // free slot -> starts immediately (processing panel)
  } else {
    await editQueuePanel(entry); // busy -> show live queue panel
  }
}

// ---- Support ticket system --------------------------------------------------
// The support panel is never posted automatically: it only appears when an
// allowed manager (SUPPORT_PANEL_MANAGER_IDS) runs `/set module: Support
// Panel` in the channel where the panel should live. Only members who hold
// SUPPORT_ROLE_ID (or any role positioned above it in the role hierarchy) may
// press its "Support" button, pick a topic, and a PRIVATE thread opens that
// only the ticket owner can see and write in. The first
// (top) message of the thread is a panel that says to wait
// for a moderator, pings SUPPORT_NOTIFY_MOD_ID + the owner, and carries a
// "Close Ticket" button that deletes the thread. Runtime state (panel message
// id + open tickets) is persisted to supportData.json so restarts never
// duplicate a panel or lose track of open tickets.
const SUPPORT_PANEL_MANAGER_IDS = new Set(["1523184178567581817", "1280789307027755019"]); // who may run "/set module: Support Panel"
const SUPPORT_ROLE_ID = "1522552781268058122"; // role that may press "Support" — or any role above it (hierarchy)
// Staff role IDs used for the "Close Ticket" permission check. Members are no
// longer auto-added to tickets (Discord rejects role IDs on that endpoint).
const SUPPORT_STAFF_IDS = [
  // staff roles that may close tickets
  "1523184178567581817",
  "1522552841716371606",
  "1522552815162097774",
  "1523629873069822022",
  "1544956355398336565",
];
const SUPPORT_NOTIFY_MOD_ID = "1523184178567581817"; // role (or user) pinged at the top of each ticket

// Ticket topics shown in the dropdown when someone presses "Support". Each has
// its own accent color used across the ticket UI.
const SUPPORT_CATEGORIES = [
  { value: "general", label: "General Help", emoji: "💬", description: "Questions about the server or bot", color: 0x3b82f6 },
  { value: "technical", label: "Technical Issue", emoji: "🔧", description: "Bugs, errors, technical problems", color: 0xf59e0b },
  { value: "other", label: "Other", emoji: "📦", description: "Anything else", color: 0x8b5cf6 },
];

function categoryByValue(value) {
  return SUPPORT_CATEGORIES.find((c) => c.value === value) || null;
}

// Short, human-friendly ticket code shown in the ticket and confirmations
// (e.g. T-7K2MXQ). No confusing 0/O or 1/I characters.
function generateTicketId() {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let id = "";
  for (let i = 0; i < 6; i++) id += chars[Math.floor(Math.random() * chars.length)];
  return `T-${id}`;
}

function formatDuration(ms) {
  const totalMinutes = Math.floor(ms / 60000);
  const days = Math.floor(totalMinutes / 1440);
  const hours = Math.floor((totalMinutes % 1440) / 60);
  const minutes = totalMinutes % 60;
  const parts = [];
  if (days) parts.push(`${days}d`);
  if (hours) parts.push(`${hours}h`);
  if (minutes) parts.push(`${minutes}m`);
  return parts.join(" ") || "< 1m";
}

// A configured ID may be a user ID or a role ID — match either one. A member
// qualifies when they ARE the user or they HOLD the role.
function memberMatchesAny(member, ids) {
  if (!member) return false;
  const roles = member.roles?.cache;
  return (ids || []).some((id) => member.id === id || (roles && roles.has(id)));
}
const SUPPORT_COLOR = 0x3b82f6;
const SUPPORT_DATA_FILE = path.join(__dirname, "supportData.json");

const supportState = loadSupportState();

function loadSupportState() {
  try {
    const parsed = JSON.parse(fs.readFileSync(SUPPORT_DATA_FILE, "utf8"));
    return {
      panel: parsed?.panel || null,
      tickets:
        parsed?.tickets && typeof parsed.tickets === "object" ? parsed.tickets : {},
    };
  } catch {
    return { panel: null, tickets: {} }; // no file yet or unreadable -> start empty
  }
}

function saveSupportState() {
  try {
    fs.writeFileSync(SUPPORT_DATA_FILE, JSON.stringify(supportState, null, 2));
  } catch (error) {
    console.error("⚠️ Failed to save support ticket state:", error.message);
  }
}

// Look up a user's open ticket thread, cleaning up the stored record
// automatically when the thread no longer exists.
async function findOpenTicket(guildId, userId) {
  const record = supportState.tickets?.[guildId]?.[userId];
  if (!record) return null;
  const thread = await client.channels.fetch(record.threadId, { force: true }).catch(() => null);
  if (!thread) {
    delete supportState.tickets[guildId][userId];
    if (Object.keys(supportState.tickets[guildId]).length === 0) {
      delete supportState.tickets[guildId];
    }
    saveSupportState();
    return null;
  }
  return thread;
}

// Discord forbids a few characters in thread names, so scrub them from the
// username — a weird name can never make thread creation fail.
function supportThreadName(user, category) {
  const clean = (user.username || "User")
    .replace(/[\u0000-\u001F\u007F<>:"\/\\|?*#@]/g, "")
    .trim();
  const prefix = category ? `${category.emoji} ${category.label}` : "📞 Support";
  return `${prefix} · ${clean || "User"}`.slice(0, 100);
}

// Build the reusable support panel content (embed + "Support"/"FAQ" buttons).
function supportPanelPayload() {
  const panelEmbed = new EmbedBuilder()
    .setColor(SUPPORT_COLOR)
    .setTitle("🛟 Support Center")
    .setDescription(
      "Need help or have a question? We're here for you — and it only takes a moment to get started."
    )
    .addFields(
      {
        name: "📩 How to get help",
        value:
          "Press the **Support** button, pick a topic, and a **private ticket** opens automatically. A moderator will reply as soon as possible.",
      },
      { name: "🕐 Response time", value: "Usually within a few hours.", inline: true },
      { name: "🔒 Privacy", value: "Only you and the support team can see your ticket.", inline: true },
      { name: "❓ Quick answers", value: "Press **FAQ** below for common questions.", inline: true }
    )
    .setFooter({ text: "Please only open a ticket when you actually need help." });
  const panelRow = new ActionRowBuilder().addComponents(
    new ButtonBuilder()
      .setCustomId("support_open")
      .setLabel("Support")
      .setEmoji("📩")
      .setStyle(ButtonStyle.Primary),
    new ButtonBuilder()
      .setCustomId("support_faq")
      .setLabel("FAQ")
      .setEmoji("❓")
      .setStyle(ButtonStyle.Secondary)
  );
  return { embeds: [panelEmbed], components: [panelRow] };
}

// "FAQ" button on the panel -> a quick ephemeral list of common questions.
async function handleSupportFaq(interaction) {
  const faqEmbed = new EmbedBuilder()
    .setColor(SUPPORT_COLOR)
    .setTitle("❓ Frequently Asked Questions")
    .addFields(
      {
        name: "🕐 How fast will I get a reply?",
        value: "Most tickets are answered within a few hours. You'll be pinged inside your ticket when a moderator responds.",
      },
      {
        name: "🔒 Is my ticket private?",
        value: "Yes. Your ticket is a private thread that only you and the support team can see.",
      },
      {
        name: "📎 Can I attach files?",
        value: "Absolutely — screenshots, logs, and other files help us solve your issue faster.",
      },
      {
        name: "🔐 How do I close my ticket?",
        value: "Press the **Close Ticket** button at the top of your ticket when you're done.",
      }
    )
    .setFooter({ text: "Still stuck? Press Support to open a ticket." });
  await interaction.reply({ embeds: [faqEmbed], ephemeral: true });
}

// True when a message is one of our support panels.
function isSupportPanelMessage(message) {
  return (
    message &&
    message.components.some((row) =>
      row.components.some((button) => button.customId === "support_open")
    )
  );
}

// Look for the panel in a channel: first the recorded message id, then a scan
// of the recent history (covers a lost supportData.json record).
async function findPanelInChannel(channel) {
  const stored = supportState.panel;
  const cached =
    stored && stored.channelId === channel.id
      ? await channel.messages.fetch(stored.messageId).catch(() => null)
      : null;
  if (isSupportPanelMessage(cached)) return cached;
  const recent = await channel.messages.fetch({ limit: 100 }).catch(() => null);
  const found =
    recent && recent.find((message) => isSupportPanelMessage(message));
  return found || null;
}

// Post the support panel in a channel (used by "/set module: Support Panel").
// Never duplicates: reuses the panel already in that channel, and when the
// panel is moved to a different channel the old one is deleted first.
async function ensureSupportPanel(channel) {
  const guild = channel.guild;

  // 1. Already in this channel? (recorded message id, then recent history)
  const existing = await findPanelInChannel(channel);
  if (existing) {
    supportState.panel = { guildId: guild.id, channelId: channel.id, messageId: existing.id };
    saveSupportState();
    console.log(`🛟 Support panel already present in #${channel.name} (message ${existing.id}).`);
    return `✅ The **Support Panel** is already posted in ${channel} — I reused it.`;
  }

  // 2. The panel currently lives in another channel? Remove the old one so
  //    there is always exactly one live panel.
  const oldRecord = supportState.panel;
  if (oldRecord && oldRecord.channelId && oldRecord.channelId !== channel.id) {
    const oldChannel = await guild.channels.fetch(oldRecord.channelId).catch(() => null);
    const oldMessage = oldChannel
      ? await oldChannel.messages.fetch(oldRecord.messageId).catch(() => null)
      : null;
    if (isSupportPanelMessage(oldMessage)) {
      await oldMessage.delete("Support panel moved to another channel");
      console.log(`🗑️ Removed old support panel from #${oldChannel.name}.`);
    }
  }

  // 3. Post a fresh panel in this channel.
  const sent = await channel.send(supportPanelPayload());
  supportState.panel = { guildId: guild.id, channelId: channel.id, messageId: sent.id };
  saveSupportState();
  console.log(`🛟 Support panel posted in #${channel.name} (message ${sent.id}).`);
  return `✅ **Support Panel** posted in ${channel}! Users can now press **Support** to open a private support ticket.`;
}

// On startup: drop ticket records whose threads no longer exist (deleted while
// the bot was offline). Does NOT post the panel — that only happens via /set.
async function pruneStaleTickets() {
  for (const [guildId, byUser] of Object.entries(supportState.tickets)) {
    for (const [userId, record] of Object.entries(byUser)) {
      const thread = await client.channels.fetch(record.threadId).catch(() => null);
      if (!thread) {
        delete supportState.tickets[guildId][userId];
        console.log(`🧹 Removed stale support ticket record for ${userId} (${record.threadId})`);
      }
    }
    if (Object.keys(supportState.tickets[guildId]).length === 0) {
      delete supportState.tickets[guildId];
    }
  }
  saveSupportState();
}

// Guards against two rapid clicks ever creating two threads for one user, and
// two concurrent /set calls ever posting two panels in the same channel.
const ticketCreationLocks = new Map();
const panelSetupLocks = new Map();

// Support-role cache (per guild, refreshed every 5 minutes) so every button
// press does not hit the Discord API, and the role's position stays stable.
const anchorRoleCache = new Map(); // guildId -> { role, fetchedAt }
const ANCHOR_ROLE_CACHE_TTL_MS = 5 * 60 * 1000;

async function getAnchorRole(guild) {
  const cached = anchorRoleCache.get(guild.id);
  if (cached && Date.now() - cached.fetchedAt < ANCHOR_ROLE_CACHE_TTL_MS) return cached.role;
  const role = await guild.roles.fetch(SUPPORT_ROLE_ID);
  anchorRoleCache.set(guild.id, { role, fetchedAt: Date.now() });
  return role;
}

// May this member press the "Support" button? They qualify when they hold
// SUPPORT_ROLE_ID, or when any role they hold sits above it in the role
// hierarchy. If the support role doesn't exist in the guild, nobody can
// qualify. Every decision is logged so a denied press can be diagnosed.
async function canOpenSupportTicket(member) {
  try {
    const anchorRole = await getAnchorRole(member.guild);
    if (member.roles.cache.has(SUPPORT_ROLE_ID)) {
      console.log(`🎟️ ${member.user.tag} (${member.id}) allowed: holds the support role.`);
      return true;
    }
    const allowed = member.roles.cache.some((role) => role.position > anchorRole.position);
    console.log(
      `🎟️ ${member.user.tag} (${member.id}) ${allowed ? "allowed" : "DENIED"}: ` +
        `highest role "${member.roles.highest.name}" pos ${member.roles.highest.position} vs ` +
        `support role "${anchorRole.name}" pos ${anchorRole.position}.`
    );
    return allowed;
  } catch (error) {
    console.error(
      `🎟️ ${member.user.tag} (${member.id}) DENIED in guild "${member.guild.name}" (${member.guild.id}): ` +
        `could not fetch support role ${SUPPORT_ROLE_ID} — ${error.message}`
    );
    return false;
  }
}

// "Support" button on the panel -> check permission, then ask which topic the
// ticket is about (the actual ticket opens once the topic is picked).
async function handleSupportOpen(interaction) {
  const user = interaction.user;
  // interaction.member is normally included on guild button presses; fall back
  // to a REST fetch just in case it is ever missing.
  const member =
    interaction.member || (await interaction.guild.members.fetch(user.id).catch(() => null));
  if (!member || !(await canOpenSupportTicket(member))) {
    console.log(`🎟️ DENIED press by ${user.tag} (${user.id}) in guild ${interaction.guild?.id}.`);
    await interaction.reply({
      content: "❌ You are not allowed to open a support ticket.",
      ephemeral: true,
    });
    return;
  }
  const channel = interaction.channel;
  if (!channel || !channel.guild || typeof channel.threads?.create !== "function") {
    await interaction.reply({
      content: "❌ Support tickets can't be opened in this channel.",
      ephemeral: true,
    });
    return;
  }

  // Ask which topic the ticket is about before opening anything. If a ticket
  // is already open, the user finds out right after picking a topic.
  const topicRow = new ActionRowBuilder().addComponents(
    new StringSelectMenuBuilder()
      .setCustomId("support_category")
      .setPlaceholder("Choose a topic for your ticket")
      .addOptions(
        SUPPORT_CATEGORIES.map((c) => ({
          label: `${c.emoji} ${c.label}`,
          value: c.value,
          description: c.description,
        }))
      )
  );
  await interaction.reply({
    content:
      "🛟 **What can we help you with?** Choose a topic below to open your private support ticket.",
    components: [topicRow],
    ephemeral: true,
  });
}

// Topic picked in the dropdown -> open the ticket for that category.
async function handleSupportCategory(interaction) {
  const category = categoryByValue(interaction.values?.[0]) || SUPPORT_CATEGORIES[0];
  await openSupportTicket(interaction, category);
}

async function openSupportTicket(interaction, category) {
  const user = interaction.user;
  await interaction.deferReply({ ephemeral: true });

  const lockKey = `${interaction.guildId}:${user.id}`;
  if (ticketCreationLocks.has(lockKey)) {
    await interaction.editReply({
      content: "⏳ Your ticket is already being opened — one moment.",
    });
    return;
  }
  ticketCreationLocks.set(lockKey, true);
  try {
    // Never create a second ticket while one is already open.
    const existing = await findOpenTicket(interaction.guildId, user.id);
    if (existing) {
      await interaction.editReply({
        content: `✅ You already have an open ticket: <#${existing.id}> — a moderator will help you there.`,
      });
      return;
    }

    // Private thread: nobody can see it until they are explicitly added. Use
    // the longest auto-archive window the guild's boost tier allows, so a
    // ticket that is quiet for a while does not vanish on its owner.
    const premiumTier = interaction.guild?.premiumTier ?? 0;
    const autoArchiveDuration =
      premiumTier >= 2 ? 10080 : premiumTier >= 1 ? 4320 : 1440; // min 1w / 3d / 24h
    const thread = await interaction.channel.threads.create({
      name: supportThreadName(user, category),
      type: ChannelType.PrivateThread,
      invitable: false, // only members explicitly added below (or Manage Threads) can join
      autoArchiveDuration,
      reason: `Support ticket opened by ${user.tag} (${category.label})`,
    });

    // Remember the ticket before anything else, so even a partial failure below
    // never lets the same user end up with two tickets.
    const ticketId = generateTicketId();
    if (!supportState.tickets[interaction.guildId]) supportState.tickets[interaction.guildId] = {};
    supportState.tickets[interaction.guildId][user.id] = {
      threadId: thread.id,
      username: user.username,
      openedAt: Date.now(),
      category: category.value,
      ticketId,
    };
    saveSupportState();

    const ticketEmbed = new EmbedBuilder()
      .setColor(category.color)
      .setAuthor({ name: user.username, iconURL: user.displayAvatarURL({ size: 64 }) })
      .setTitle(`📞 ${category.emoji} ${category.label}`)
      .setDescription(
        "Thank you for contacting support! Please describe your issue in as much detail as possible below.\n\n" +
          `🆔 **Ticket:** \`${ticketId}\`\n` +
          `🗂 **Category:** ${category.emoji} ${category.label}\n` +
          `📌 **Status:** Please wait for a moderator to respond.\n` +
          `🕐 **Opened:** <t:${Math.floor(Date.now() / 1000)}:R>`
      )
      .setFooter({ text: "Close the ticket with the button below when you're done." });
    const closeRow = new ActionRowBuilder().addComponents(
      new ButtonBuilder()
        .setCustomId(`support_close:${user.id}`)
        .setLabel("Close Ticket")
        .setEmoji("🔒")
        .setStyle(ButtonStyle.Danger)
    );

    // The top (first) message of the thread: a panel that says to wait for a
    // moderator and pings the moderator + the ticket owner.
    await thread.send({
      content: `**Please wait for a moderator to respond** to your ticket.\n<@&${SUPPORT_NOTIFY_MOD_ID}> <@${user.id}>`,
      embeds: [ticketEmbed],
      components: [closeRow],
    });

    console.log(
      `📞 Support ticket created: "${thread.name}" (${thread.id}) by ${user.tag} [${category.label}]`
    );

    // Clean confirmation: an embed instead of plain text.
    const confirmEmbed = new EmbedBuilder()
      .setColor(category.color)
      .setTitle("✅ Ticket Opened")
      .setDescription(`Your **${category.label}** ticket is ready: <#${thread.id}>`)
      .addFields(
        { name: "🆔 Ticket", value: `\`${ticketId}\``, inline: true },
        { name: "🗂 Category", value: `${category.emoji} ${category.label}`, inline: true },
        { name: "🕐 Opened", value: `<t:${Math.floor(Date.now() / 1000)}:R>`, inline: true }
      )
      .setFooter({ text: "Please wait for a moderator to respond in your ticket." });

    await interaction.editReply({ embeds: [confirmEmbed] });
  } catch (error) {
    console.error("❌ Failed to open support ticket:", error.message);
    await interaction
      .editReply({
        content:
          `❌ Failed to open a support ticket: ${error.message}\n` +
          "Make sure the bot has the **Create Private Threads** and **Manage Threads** " +
          "permissions in the support channel, then try again.",
      })
      .catch(() => {});
  } finally {
    ticketCreationLocks.delete(lockKey);
  }
}

async function handleSupportClose(interaction) {
  const thread = interaction.channel;
  const creatorId = (interaction.customId || "").split(":")[1];
  const isOwner = creatorId === interaction.user.id;
  const isStaff = memberMatchesAny(interaction.member, SUPPORT_STAFF_IDS);
  const canManageThreads =
    interaction.memberPermissions?.has(PermissionFlagsBits.ManageThreads) ||
    interaction.memberPermissions?.has(PermissionFlagsBits.Administrator);

  if (!thread || !thread.guild || thread.type !== ChannelType.PrivateThread) {
    await interaction.reply({
      content: "❌ This button can only be used inside a support ticket.",
      ephemeral: true,
    });
    return;
  }
  if (!isOwner && !isStaff && !canManageThreads) {
    await interaction.reply({
      content: "❌ You don't have permission to close this ticket.",
      ephemeral: true,
    });
    return;
  }

  await interaction.deferReply({ ephemeral: true });
  try {
    // Snapshot the record, then forget the ticket first, so a new one can be
    // opened right away.
    const record = creatorId ? supportState.tickets[thread.guildId]?.[creatorId] : null;
    if (creatorId && supportState.tickets[thread.guildId]?.[creatorId]) {
      delete supportState.tickets[thread.guildId][creatorId];
      if (Object.keys(supportState.tickets[thread.guildId]).length === 0) {
        delete supportState.tickets[thread.guildId];
      }
      saveSupportState();
    }

    await interaction.editReply({ content: "🔒 Closing this ticket…" });

    const category = record ? categoryByValue(record.category) : null;
    const closedEmbed = new EmbedBuilder()
      .setColor(0x6b7280)
      .setTitle("🔒 Ticket Closed")
      .setDescription(
        `Your support ticket **${thread.name}** has been closed. Thanks for reaching out!`
      )
      .addFields(
        { name: "🆔 Ticket", value: `\`${record?.ticketId || "—"}\``, inline: true },
        {
          name: "🗂 Category",
          value: category ? `${category.emoji} ${category.label}` : "—",
          inline: true,
        },
        {
          name: "⏱ Duration",
          value: record ? formatDuration(Date.now() - record.openedAt) : "—",
          inline: true,
        }
      )
      .setFooter({ text: "Feel free to open a new ticket anytime if you need more help." });

    try {
      await thread.delete(`Support ticket closed by ${interaction.user.tag}`);
      console.log(`🗑️ Support ticket ${thread.id} closed by ${interaction.user.tag}`);
      await interaction.editReply({ embeds: [closedEmbed] });
    } catch (deleteError) {
      // No permission to delete -> archive & lock as a fallback so the
      // conversation is still closed and can't be written in anymore.
      console.error(
        `⚠️ Could not delete ticket ${thread.id}: ${deleteError.message} — archiving instead.`
      );
      try {
        await thread.setLocked(true);
        await thread.setArchived(true);
      } catch (_) {
        /* ignore secondary failures */
      }
      await interaction
        .editReply({
          content:
            "⚠️ The ticket could not be deleted, so it was archived instead. " +
            "A moderator can delete it manually.",
        })
        .catch(() => {});
    }
  } catch (error) {
    console.error("❌ Failed to close support ticket:", error.message);
    await interaction
      .editReply({ content: "❌ Failed to close the ticket. Please try again." })
      .catch(() => {});
  }
}

// Bot ready event
client.once(Events.ClientReady, async () => {
  console.log(`🤖 Bot logged in as ${client.user.tag}`);
  console.log(`🧠 Nexora AI model chain: ${GEMINI_MODELS.join(" → ")}`);
  console.log(`📊 Queue config: max ${MAX_CONCURRENT_TASKS} concurrent tasks, max ${MAX_WAITING_JOBS} waiting`);
  console.log(`📡 Latency probe: every ${LATENCY_SAMPLE_INTERVAL_MS / 1000}s, rolling ${LATENCY_HISTORY_MAX} samples`);
  updatePresence();
  sampleLatencyOnce().then(updatePresence); // measure immediately on startup
  setInterval(() => {
    sampleLatencyOnce().then(updatePresence);
  }, LATENCY_SAMPLE_INTERVAL_MS);
  await registerCommands();
  // Clean up ticket records whose threads were deleted while the bot was
  // offline. The support panel itself is only posted via `/set module:
  // Support Panel` — never automatically.
  await pruneStaleTickets().catch((error) =>
    console.error("❌ Support ticket cleanup failed:", error.message)
  );
});

// Handle slash command interactions
client.on("interactionCreate", async (interaction) => {
  // "Refresh" button on the /status panel -> re-measure everything live.
  if (interaction.isButton() && interaction.customId === "status_refresh") {
    try {
      await interaction.deferUpdate();
      await sampleLatencyOnce(); // force a fresh reading before rebuilding
      updatePresence();
      await interaction.editReply({
        embeds: [await statusEmbed()],
        components: [statusActionRow()],
      });
    } catch (error) {
      console.error("❌ Error refreshing /status:", error);
    }
    return;
  }

  // Support ticket buttons: "Support" + "FAQ" on the panel and "Close Ticket"
  // inside a ticket (private thread).
  if (interaction.isButton()) {
    if (interaction.customId === "support_open") {
      await handleSupportOpen(interaction);
      return;
    }
    if (interaction.customId === "support_faq") {
      await handleSupportFaq(interaction);
      return;
    }
    if (interaction.customId.startsWith("support_close:")) {
      await handleSupportClose(interaction);
      return;
    }
  }

  // Ticket topic dropdown picked on the panel -> open the ticket for that topic.
  if (interaction.isStringSelectMenu() && interaction.customId === "support_category") {
    await handleSupportCategory(interaction);
    return;
  }

  if (!interaction.isChatInputCommand()) return;

  if (interaction.commandName === "status") {
    try {
      await interaction.reply({
        embeds: [await statusEmbed()],
        components: [statusActionRow()],
      });
    } catch (error) {
      console.error("❌ Error handling /status:", error);
    }
    return;
  }

  if (interaction.commandName === "set") {
    try {
      const module = interaction.options.getString("module");

      if (module === "off") {
        aiChannels.delete(interaction.guildId);
        saveAiChannels();
        await interaction.reply({
          content: "🛑 **Nexora AI** has been turned off in this server. I'll stop replying here.",
        });
        return;
      }

      // Post the support panel in this channel (only allowed managers).
      if (module === "support_panel") {
        if (!memberMatchesAny(interaction.member, SUPPORT_PANEL_MANAGER_IDS)) {
          await interaction.reply({
            content: "❌ You are not allowed to set up the support panel.",
            ephemeral: true,
          });
          return;
        }
        const channel = interaction.channel;
        if (!channel || !channel.guild || channel.type !== ChannelType.GuildText) {
          await interaction.reply({
            content: "❌ The support panel can only be posted in a text channel.",
            ephemeral: true,
          });
          return;
        }
        await interaction.deferReply();
        const panelLockKey = `${interaction.guildId}:${channel.id}`;
        if (panelSetupLocks.has(panelLockKey)) {
          await interaction.editReply({
            content: "⏳ The support panel is already being set up — one moment.",
          });
          return;
        }
        panelSetupLocks.set(panelLockKey, true);
        try {
          const message = await ensureSupportPanel(channel);
          await interaction.editReply({ content: message });
        } catch (error) {
          console.error("❌ Failed to post the support panel:", error.message);
          await interaction.editReply({
            content:
              "❌ Failed to post the support panel. Check my permissions in this channel " +
              "(Send Messages) and try again.",
          });
        } finally {
          panelSetupLocks.delete(panelLockKey);
        }
        return;
      }

      if (!process.env.GEMINI_API_KEY) {
        await interaction.reply({
          content:
            "❌ **GEMINI_API_KEY** is not set in `.env`. Add your Google Gemini API key, " +
            "restart the bot, then run `/set` again.",
          ephemeral: true,
        });
        return;
      }

      aiChannels.set(interaction.guildId, interaction.channelId);
      saveAiChannels();
      await interaction.reply({
        content:
          `✅ **Nexora AI** is now active in ${interaction.channel}! ` +
          "I'll reply to every message sent here. Use `/set module: Off` to disable it.",
      });
    } catch (error) {
      console.error("❌ Error handling /set:", error);
      await interaction
        .reply({
          content: "❌ Something went wrong while setting the module. Please try again.",
          ephemeral: true,
        })
        .catch(() => {});
    }
    return;
  }

  if (interaction.commandName === "newtask") {
    const cleared = deleteConversation(interaction.guildId);
    await interaction.reply({
      content: cleared
        ? "🧹 **Nexora AI** memory cleared — starting a fresh conversation!"
        : "🧹 Nexora AI memory is already empty — nothing to forget.",
    });
    return;
  }

  if (interaction.commandName === "analyze") {
    await handleAnalyzeCommand(interaction);
    return;
  }

  if (interaction.commandName === "imagine") {
    await handleImagineCommand(interaction);
    return;
  }

  if (interaction.commandName !== "removebg") return;

  try {
    // Acknowledge interaction FIRST to prevent 3-second timeout
    await interaction.deferReply();

    const attachment = interaction.options.getAttachment("image");

    // Validate file type
    const validTypes = ["image/png", "image/jpeg", "image/webp"];
    if (!attachment || !attachment.contentType || !validTypes.includes(attachment.contentType)) {
      await interaction.editReply({
        content: "❌ Please provide a valid image (PNG, JPEG, or WebP)!",
      });
      return;
    }

    // Check file size (max 8MB for processing)
    if (attachment.size > MAX_FILE_SIZE) {
      await interaction.editReply({
        content: "❌ Image too large! Maximum size is 8MB.",
      });
      return;
    }

    // Queue the request (FIFO, max 20 concurrent). If the queue is busy the
    // user immediately gets a live panel showing their position.
    await enqueueRequest(interaction, attachment);
  } catch (error) {
    console.error("❌ Error handling interaction:", error);
    try {
      if (interaction.deferred || interaction.replied) {
        await interaction.editReply({ content: friendlyError(error) });
      } else {
        await interaction.reply({ content: friendlyError(error), ephemeral: true });
      }
    } catch (replyError) {
      console.error("❌ Failed to send error reply:", replyError.message);
    }
  }
});

// ---- Bot commands spoken in plain chat --------------------------------------
// In an AI channel the real bot commands can be run by just typing them:
// "status" shows the live status panel, "newtask"/"reset" clears the chat
// memory, "help"/"bantuan" lists everything. Patterns are deliberately strict
// so normal conversation (e.g. "apa itu status error?") is never hijacked.
const HELP_TEXT =
  "📋 **Available Commands:**\n" +
  "`/removebg` - Remove background from an image\n" +
  "`/status` - Check bot status (latency, CPU, RAM, uptime)\n" +
  "`/set` - Turn this channel into a Nexora AI chat channel or post the Support Panel\n" +
  "`/newtask` - Forget the current Nexora AI chat and start fresh\n" +
  "`/analyze` - Analyze any file (image, mp4, mp3, PDF, Word, PPTX, Excel, code...)\n" +
  "`/imagine` - Generate an image from a description; attach a reference image to edit it\n" +
  "💡 In an AI chat channel (`/set`) you can also just attach a file with your " +
  "message and Nexora will analyze it, write e.g. `buatkan gambar kucing` to " +
  "generate one, attach a photo + `ubah latarnya jadi pantai` to edit it, or " +
  "attach a photo + `hapus background` to remove its background!\n" +
  "💬 In an AI chat channel plain text also runs commands: `status`/`ping`, " +
  "`newtask`/`reset`, `riwayat` (downloads this server's chat history), " +
  "`set off`/`matikan nexora`, and `help`/`bantuan`. Paste a file link to " +
  "analyze it, type `gambar kucing` to draw one, attach a photo + `hapus " +
  "background`/`buat transparan`, or `ubah latarnya jadi pantai` to edit it.";

// Match a short, command-like chat message to a real bot command, or null.
function extractChatCommand(text) {
  const t = (text || "").trim().toLowerCase().replace(/[.!,?]+$/, "");
  if (!t || t.length > 60) return null;
  if (
    /^(?:tolong\s+|bisa\s+|cek(?:in)?\s+|lihat\s+|show\s+|check\s+|bot\s+|kasih\s+)?(?:status|stats?|statistik|keadaan)(?:\s+bot)?(?:nya)?(?:\s+(?:sekarang|dong|yuk|dulu))?$/.test(t) ||
    /^(?:cek\s+|bot\s+)?(?:ping|latensi|lag)$/.test(t)
  ) {
    return "status";
  }
  if (
    /^\/?newtask$/.test(t) ||
    /^\/?(?:reset(?:\s+(?:percakapan|chat|riwayat|memory))?|mulai\s+baru|start\s+fresh|hapus\s+riwayat|lupakan\s+percakapan|clear\s+(?:chat|memory|percakapan|riwayat)|bersihkan\s+(?:memori|chat|percakapan)|buang\s+memori)$/.test(t)
  ) {
    return "newtask";
  }
  if (
    /^(?:set\s+off|off|matikan\s+nexora|matikan\s+ai|nexora\s+off|nonaktifkan\s+(?:nexora|ai)|turn\s+off\s+(?:nexora|ai))$/.test(t)
  ) {
    return "setoff";
  }
  if (
    /^(?:riwayat(?:(?:\s+(?:chat|percakapan)))?|log\s+chat|chat\s+history|history|unduh\s+riwayat|download\s+(?:chat\s+)?history|ambil\s+riwayat)$/.test(t)
  ) {
    return "history";
  }
  if (/^(?:help|bantuan|command|commands|daftar\s+command|daftar\s+perintah)$/.test(t)) {
    return "help";
  }
  return null;
}

// Execute a command the user asked for in plain text. Returns true when it
// handled the message, so the caller does not also send it to Gemini.
async function handleChatCommand(message, text) {
  const command = extractChatCommand(text);
  if (!command) return false;
  console.log(`🕹️ Executing chat command "${command}" (guild ${message.guildId})`);
  try {
    await message.channel.sendTyping();
    if (command === "status") {
      await message.reply({
        embeds: [await statusEmbed()],
        components: [statusActionRow()],
      });
    } else if (command === "newtask") {
      const cleared = deleteConversation(message.guildId);
      await message.reply(
        cleared
          ? "🧹 **Nexora AI** memory cleared — starting a fresh conversation!"
          : "🧹 Nexora AI memory is already empty — nothing to forget."
      );
    } else if (command === "setoff") {
      aiChannels.delete(message.guildId);
      saveAiChannels();
      await message.reply(
        "🛑 **Nexora AI** has been turned off in this server — I'll stop replying here. " +
          "Use `/set module: Nexora AI` in any channel to turn me back on."
      );
    } else if (command === "history") {
      const history = loadConversation(message.guildId);
      if (history.length === 0) {
        await message.reply(
          "📭 No conversation history saved for this server yet — chat with me first!"
        );
      } else {
        const transcript = history
          .map(
            (entry) =>
              `${entry.role === "user" ? "👤 User" : "🤖 Nexora"}: ${entry.text}`
          )
          .join("\n\n──────────────────\n\n");
        const attachment = new AttachmentBuilder(
          Buffer.from(transcript, "utf8"),
          { name: `nexora_history_${message.guildId}.txt` }
        );
        await message.reply({
          content: `📜 Here's the saved conversation history for this server (${history.length} messages).`,
          files: [attachment],
        });
      }
    } else if (command === "help") {
      await message.reply(HELP_TEXT);
    }
  } catch (error) {
    console.error(`❌ Chat command "${command}" failed:`, error.message);
    await message
      .reply(`⚠️ Failed to run \`${command}\` — please try again.`)
      .catch(() => {});
  }
  return true;
}

// Send a Gemini chat answer to a message, remembering both sides on disk
// (memory/<guildId>.json) and splitting long replies into multiple messages.
// Shared by attached-file analysis and URL analysis.
async function storeAndSendChatReply(message, guildId, text, attachments, reply, model) {
  const history = loadConversation(guildId);
  const userText =
    (text || "").trim() ||
    (attachments.length > 0
      ? `(asked me to look at ${attachments.length} attached file${attachments.length > 1 ? "s" : ""})`
      : "");
  history.push({ role: "user", text: userText }, { role: "model", text: reply });
  while (history.length > MAX_HISTORY_MESSAGES) history.shift();
  saveConversation(guildId, history);
  if (model !== GEMINI_MODELS[0]) {
    console.log(`🤖 Nexora AI answered with fallback model: ${model}`);
  }
  const chunks = splitLongText(reply);
  await message.reply(chunks[0]);
  for (const chunk of chunks.slice(1)) {
    await message.channel.send(chunk);
  }
}

// ---- Analyzing files from plain URLs ----------------------------------------
// Pasting a link to an image/PDF/video/etc. (instead of attaching) works too:
// the file is downloaded and analyzed just like an attachment.
const ANALYZE_LINK_INTENT =
  /(?:analis|analy[sz]|ringkas|summari[sz]|jelaskan|baca(?:kan)?|lihat(?:in)?|cek|check|apa\s+isi|terjemahkan|translate|buka|open)/i;

function isOnlyUrls(text) {
  return /^(?:https?:\/\/\S+\s*)+$/i.test((text || "").trim());
}

// Find URLs in a message that point directly to analyzable files
// (image/audio/video/document/code extensions).
function extractFileUrls(text) {
  const matches = (text || "").match(/https?:\/\/[^\s<>"']+/gi) || [];
  const found = [];
  for (let raw of matches) {
    raw = raw.replace(/[)\]}>.,;]+$/, "");
    try {
      const pathname = new URL(raw).pathname;
      const name = decodeURIComponent(pathname.split("/").pop() || "");
      const ext = path.extname(name).slice(1).toLowerCase();
      if (name && (TEXT_FILE_EXTENSIONS.has(ext) || EXTENSION_MIME[ext])) {
        found.push({ name, url: raw });
      }
    } catch {
      /* ignore malformed URLs */
    }
  }
  return found;
}

// Treat a message as a "analyze this link" request when it contains file URLs
// and either an analysis word or is nothing but URLs.
function looksLikeAnalyzeLinkRequest(text) {
  const t = (text || "").trim();
  if (!t || t.length > 1500) return false;
  return extractFileUrls(t).length > 0 && (ANALYZE_LINK_INTENT.test(t) || isOnlyUrls(t));
}

// Download and analyze files referenced by URL(s) in a chat message.
async function analyzeFileLinksInChat(message, text) {
  const refs = extractFileUrls(text);
  if (refs.length === 0) return false;
  try {
    // Real size is unknown until download, so report 0 here and let
    // prepareAttachmentParts double-check the downloaded bytes against the limit.
    const pseudoAttachments = refs.map((ref) => ({
      name: ref.name,
      size: 0,
      contentType:
        EXTENSION_MIME[path.extname(ref.name).slice(1).toLowerCase()] ||
        "application/octet-stream",
      url: ref.url,
    }));
    console.log(`🔗 Analyzing ${refs.length} file URL(s)...`);
    const { reply, model } = await askGemini(message.guildId, text, {
      attachments: pseudoAttachments,
    });
    await storeAndSendChatReply(
      message,
      message.guildId,
      text,
      pseudoAttachments,
      reply,
      model
    );
    return true;
  } catch (error) {
    console.error("❌ URL analysis error:", error.message);
    const message_ = error.message || String(error);
    await message
      .reply(
        error.isAnalysis || message_.startsWith("❌")
          ? message_
          : `❌ Could not analyze that link: ${message_}`
      )
      .catch(() => {});
    return true;
  }
}

// Handle prefix commands (optional - for simplicity)
client.on("messageCreate", async (message) => {
  if (message.author.bot) return;

  // Simple help command
  if (message.content === "!help") {
    return message.reply(HELP_TEXT);
  }

  // Nexora AI: reply to every message in a registered chat channel. Text-only
  // messages get a Gemini reply; messages with attachments get the file(s)
  // analyzed too (images, video, audio, PDFs, Office documents, source code...).
  const aiChannelId = aiChannels.get(message.guildId);
  if (aiChannelId && message.channelId === aiChannelId) {
    const text = message.content.trim();
    const attachments = [...message.attachments.values()];
    if (!text && attachments.length === 0) return;
    try {
      await message.channel.sendTyping();

      // "remove bg" / "hapus background" + attached image -> run the exact
      // same pipeline as /removebg instead of an AI answer.
      if (text && attachments.length > 0 && looksLikeRemoveBgRequest(text)) {
        const bgImage = attachments.find(isBgRemovableAttachment);
        if (!bgImage) {
          await message.reply(
            "❌ To remove a background, attach an image (PNG, JPEG, or WebP) up to 8MB together with your request."
          );
          return;
        }
        await runChatBackgroundRemoval(message, bgImage);
        return;
      }

      // Real bot commands spoken as plain text: "status", "newtask", "help",
      // "riwayat", "set off", ...
      if (await handleChatCommand(message, text)) return;

      // Files pasted as URLs ("analisis link ini ..." or a bare file link)
      // are downloaded and analyzed just like attachments.
      if (attachments.length === 0 && looksLikeAnalyzeLinkRequest(text)) {
        if (await analyzeFileLinksInChat(message, text)) return;
      }

      // Image requests ("buatkan gambar ...", "gambar kucing", "/imagine ...",
      // ...) are routed to the dedicated image-generation model. If images are
      // attached they are used as references, so the prompt becomes an editing
      // instruction (e.g. attach a photo + "ubah latarnya jadi pantai").
      const wantsImageEdit =
        attachments.length > 0 && looksLikeImageEditRequest(text);
      const wantsImageGen =
        looksLikeImageRequest(text) || looksLikeBareImageRequest(text);
      if (text && (wantsImageGen || wantsImageEdit)) {
        const prompt = text.replace(/^\/imagine\b/, "").trim() || text;
        const referenceParts = attachments.length > 0 ? await prepareImageReferenceParts(attachments) : [];
        const { buffer, mime, model } = await generateImageFromPrompt(prompt, referenceParts);
        const file = await attachmentFromImageBuffer(buffer, mime, "nexora_image");
        const caption =
          referenceParts.length > 0
            ? `🎨 Edited from your image${referenceParts.length > 1 ? "s" : ""}! **${prompt}**`
            : `🎨 Here you go! **${prompt}**`;
        await message.reply({ content: caption, files: [file] });
        if (model !== IMAGE_MODELS[0]) {
          console.log(`🎨 Nexora generated an image with fallback model: ${model}`);
        }
        return;
      }

      const { reply, model } = await askGemini(message.guildId, text, { attachments });
      // Remember both sides so the next message has full context — stored on
      // disk (memory/<guildId>.json), not RAM. Attachments are not stored —
      // only the text the user wrote.
      await storeAndSendChatReply(message, message.guildId, text, attachments, reply, model);
    } catch (error) {
      console.error("❌ Nexora AI error:", error.message);
      const message_ = error.message || String(error);
      // File-analysis and image-generation errors already carry a complete,
      // user-friendly explanation — surface them as-is.
      if (error.isAnalysis) {
        await message.reply(message_).catch(() => {});
        return;
      }
      if (error.isImage) {
        await message.reply(imageGenerationErrorMessage(error)).catch(() => {});
        return;
      }
      let hint;
      if (message_.includes("GEMINI_API_KEY")) {
        hint = "Set **GEMINI_API_KEY** in `.env`, restart the bot, then try again.";
      } else if (message_.includes("suspended") || message_.includes("CONSUMER_SUSPENDED")) {
        hint =
          "Google has suspended the project behind the **GEMINI_API_KEY** (usually a billing issue). " +
          "Check the Google Cloud console → Billing, or create a fresh key in AI Studio and update `.env`.";
      } else if (message_.includes("HTTP 403") || message_.includes("HTTP 401")) {
        hint = "Your Gemini API key was rejected — verify it is still valid in Google AI Studio.";
      } else if (message_.includes("HTTP 404") || message_.includes("no longer available")) {
        hint =
          "The Gemini model is outdated. Set **GEMINI_MODEL** / **GEMINI_MODELS** in `.env` " +
          "to a current model (e.g. `gemini-3.6-flash`), restart the bot, then try again.";
      } else if (message_.includes("All Gemini models failed")) {
        hint =
          "All configured Gemini models failed — check the **GEMINI_MODELS** in `.env` and " +
          "your API key/quota, then try again.";
      } else {
        hint = "Please try again in a moment.";
      }
      await message.reply(`⚠️ Nexora AI hit an error: ${hint}`).catch(() => {});
    }
  }
});

// Handle bot errors to prevent crashes
client.on("error", (error) => {
  console.error("❌ Bot error:", error.message);
});

process.on("unhandledRejection", (error) => {
  console.error("❌ Unhandled rejection:", error instanceof Error ? error.message : error);
});

// Login to Discord (only when run directly, so tests can require this module)
if (require.main === module) {
  client.login(process.env.DISCORD_TOKEN).catch((error) => {
    console.error("❌ Failed to log in:", error && error.message ? error.message : error);
    process.exit(1);
  });
}

module.exports = client;
