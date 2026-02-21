import { describe, expect, it, vi } from "vitest";
import { logger } from "./logger";

describe("logger", () => {
  it("logs info messages as JSON to console.info", () => {
    const spy = vi.spyOn(console, "info").mockImplementation(() => {});
    logger.info("test message", { key: "value" });
    expect(spy).toHaveBeenCalledOnce();
    const payload = JSON.parse(spy.mock.calls[0][0] as string);
    expect(payload.level).toBe("info");
    expect(payload.message).toBe("test message");
    expect(payload.context).toEqual({ key: "value" });
    expect(payload.ts).toBeTruthy();
    spy.mockRestore();
  });

  it("logs warn messages to console.warn", () => {
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    logger.warn("warning");
    expect(spy).toHaveBeenCalledOnce();
    const payload = JSON.parse(spy.mock.calls[0][0] as string);
    expect(payload.level).toBe("warn");
    spy.mockRestore();
  });

  it("logs error messages to console.error", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    logger.error("failure", { code: 500 });
    expect(spy).toHaveBeenCalledOnce();
    const payload = JSON.parse(spy.mock.calls[0][0] as string);
    expect(payload.level).toBe("error");
    expect(payload.context).toEqual({ code: 500 });
    spy.mockRestore();
  });

  it("omits context when not provided", () => {
    const spy = vi.spyOn(console, "info").mockImplementation(() => {});
    logger.info("no context");
    const payload = JSON.parse(spy.mock.calls[0][0] as string);
    expect(payload.context).toBeUndefined();
    spy.mockRestore();
  });
});
