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
  Status,
} = require("discord.js");
const axios = require("axios");
const os = require("os");
const sharp = require("sharp");
const { removeBackground } = require("@imgly/background-removal-node");

const MAX_FILE_SIZE = 8 * 1024 * 1024; // 8MB
const PROCESS_TIMEOUT_MS = 120000; // 2 minutes

// Images above this many pixels on the longest side skip the aspect-correction
// padding step, so very large images don't blow up memory on the square canvas.
const MAX_PAD_CANVAS = 4096;

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

// Download image with axios and retry logic
async function downloadImage(url, retries = 3) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      console.log(`📥 Downloading image (attempt ${attempt}/${retries})...`);
      const response = await axios.get(url, {
        responseType: "arraybuffer",
        timeout: 60000, // 60 second timeout
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
      });
      console.log("✅ Image downloaded successfully!");
      return Buffer.from(response.data);
    } catch (error) {
      console.error(`❌ Download attempt ${attempt} failed:`, error.message);
      if (attempt === retries) {
        throw new Error(`Failed to download image after ${retries} attempts: ${error.message}`);
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

// The actual image processing. Everything from download to sending the result.
async function runJob({ interaction, attachment }) {
  try {
    await interaction.editReply({ embeds: [processingPanelEmbed()] });

    // Download the image with retry
    const imageBuffer = await downloadImage(attachment.url);

    // ---- Aspect-correct inference ----
    // The library resizes every image to a 1024x1024 square before running the
    // model. For non-square photos that stretches the subject, which makes the
    // cutout less precise. Padding the input to a square canvas first keeps the
    // subject undistorted during inference; we crop back to the exact original
    // pixel dimensions afterwards, so the result keeps its original resolution.
    const metadata = await sharp(imageBuffer).metadata();
    const { width, height } = metadata;
    let inferenceBlob = new Blob([imageBuffer], { type: attachment.contentType });
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
      removeBackground(new Blob([imageBuffer], { type: attachment.contentType }), {
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

// Bot ready event
client.once(Events.ClientReady, async () => {
  console.log(`🤖 Bot logged in as ${client.user.tag}`);
  console.log(`📊 Queue config: max ${MAX_CONCURRENT_TASKS} concurrent tasks, max ${MAX_WAITING_JOBS} waiting`);
  console.log(`📡 Latency probe: every ${LATENCY_SAMPLE_INTERVAL_MS / 1000}s, rolling ${LATENCY_HISTORY_MAX} samples`);
  updatePresence();
  sampleLatencyOnce().then(updatePresence); // measure immediately on startup
  setInterval(() => {
    sampleLatencyOnce().then(updatePresence);
  }, LATENCY_SAMPLE_INTERVAL_MS);
  await registerCommands();
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

// Handle prefix commands (optional - for simplicity)
client.on("messageCreate", async (message) => {
  if (message.author.bot) return;

  // Simple help command
  if (message.content === "!help") {
    return message.reply(
      "📋 **Available Commands:**\n" +
        "`/removebg` - Remove background from an image\n" +
        "`/status` - Check bot status (latency, CPU, RAM, uptime)\n" +
        "Just use the slash command and attach an image!"
    );
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
