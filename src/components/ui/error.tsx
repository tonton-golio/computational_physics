import Link from "next/link";

interface ErrorMessageProps {
  title?: string;
  message: string;
  retry?: () => void;
}

export function ErrorMessage({ title = "Error", message, retry }: ErrorMessageProps) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-4">
      <div className="flex items-start gap-3">
        <span className="text-xl">⚠️</span>
        <div>
          <h3 className="font-semibold text-red-800">{title}</h3>
          <p className="mt-1 text-sm text-red-600">{message}</p>
          {retry && (
            <button
              onClick={retry}
              className="mt-3 rounded-lg bg-red-100 px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-200"
            >
              Try Again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function NotFoundPage({ 
  title = "Page Not Found", 
  message = "The page you're looking for doesn't exist or has been moved." 
}: { title?: string; message?: string }) {
  return (
    <div className="flex min-h-[50vh] items-center justify-center">
      <div className="text-center">
        <span className="text-6xl">🔍</span>
        <h1 className="mt-4 text-2xl font-bold text-gray-900">{title}</h1>
        <p className="mt-2 text-gray-600">{message}</p>
        <Link
          href="/"
          className="mt-6 inline-block rounded-lg bg-primary-600 px-4 py-2 text-white hover:bg-primary-700"
        >
          Go Home
        </Link>
      </div>
    </div>
  );
}
