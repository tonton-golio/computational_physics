type LogLevel = "info" | "warn" | "error";

interface LogRecord {
  level: LogLevel;
  message: string;
  context?: Record<string, unknown>;
}

function write(record: LogRecord): void {
  const payload = {
    ts: new Date().toISOString(),
    ...record,
  };

  if (record.level === "error") {
    console.error(JSON.stringify(payload));
    return;
  }
  if (record.level === "warn") {
    console.warn(JSON.stringify(payload));
    return;
  }
  console.info(JSON.stringify(payload));
}

export const logger = {
  info(message: string, context?: Record<string, unknown>) {
    write({ level: "info", message, context });
  },
  warn(message: string, context?: Record<string, unknown>) {
    write({ level: "warn", message, context });
  },
  error(message: string, context?: Record<string, unknown>) {
    write({ level: "error", message, context });
  },
};
