import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-[var(--border-strong)] bg-[var(--background)] py-6">
      <div className="w-full px-6">
        <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
          <div className="flex items-center gap-2 text-sm font-semibold text-[var(--text-strong)]">
            Computational Physics
          </div>

          <nav className="flex items-center gap-6 text-sm text-[var(--text-soft)]">
            <Link href="/" className="transition-colors hover:text-[var(--text-strong)]">Home</Link>
            <a
              href="https://github.com/tonton-golio/computational_physics"
              target="_blank"
              rel="noopener noreferrer"
              className="transition-colors hover:text-[var(--text-strong)]"
            >
              GitHub
            </a>
          </nav>

          <p className="text-sm text-[var(--text-soft)]">
            Computational Physics Learning Platform
          </p>
        </div>
      </div>
    </footer>
  );
}
