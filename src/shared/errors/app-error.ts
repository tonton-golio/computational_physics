export type AppErrorCode =
  | "BAD_REQUEST"
  | "UNAUTHORIZED"
  | "NOT_FOUND"
  | "CONTENT_READ_ERROR"
  | "CONTENT_PARSE_ERROR"
  | "INTERNAL_ERROR";

export class AppError extends Error {
  readonly code: AppErrorCode;
  readonly status: number;
  readonly details?: unknown;

  constructor(code: AppErrorCode, message: string, status = 500, details?: unknown) {
    super(message);
    this.name = "AppError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

export interface ErrorEnvelope {
  error: {
    code: AppErrorCode;
    message: string;
    details?: unknown;
  };
}

export function asErrorEnvelope(error: unknown): ErrorEnvelope {
  if (error instanceof AppError) {
    return {
      error: {
        code: error.code,
        message: error.message,
        details: error.details,
      },
    };
  }

  return {
    error: {
      code: "INTERNAL_ERROR",
      message: "An unexpected error occurred.",
    },
  };
}
