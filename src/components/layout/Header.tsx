import Link from "next/link";

export function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-gray-200 bg-white/80 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <Link href="/graph" className="flex items-center gap-2">
          <span className="text-2xl">🐨</span>
          <span className="text-xl font-bold text-gray-900">Koala-Brain</span>
        </Link>
        
        <nav className="flex items-center gap-6">
          <Link 
            href="/graph" 
            className="text-sm font-medium text-gray-600 transition-colors hover:text-gray-900"
          >
            Explore
          </Link>
          <Link 
            href="/about" 
            className="text-sm font-medium text-gray-600 transition-colors hover:text-gray-900"
          >
            About
          </Link>
        </nav>
      </div>
    </header>
  );
}
