import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-[#1e1e2e] bg-[#0a0a14] py-6">
      <div className="mx-auto max-w-7xl px-6">
        <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
          <div className="flex items-center gap-2">
            <span className="text-xl">🐨</span>
            <span className="font-semibold text-white">Koala-Brain</span>
          </div>
          
          <nav className="flex items-center gap-6 text-sm text-gray-400">
            <Link href="/graph" className="hover:text-white transition-colors">Explore</Link>
            <Link href="/about" className="hover:text-white transition-colors">About</Link>
            <a 
              href="https://github.com/tonton-golio/computational_physics" 
              target="_blank" 
              rel="noopener noreferrer"
              className="hover:text-white transition-colors"
            >
              GitHub
            </a>
          </nav>
          
          <p className="text-sm text-gray-500">
            Computational Physics Learning Platform
          </p>
        </div>
      </div>
    </footer>
  );
}
